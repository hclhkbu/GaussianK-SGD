from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import models.lstm as lstmpy


class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

    def _register_hooks(self):
        for name, p in self.model.named_parameters():
            p.register_hook(self._make_hook(name, p))

    def _make_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._backward_seq_keys) != len(self._seq_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p.numel())
            if name not in self._handles:
                self._handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._handles[name].append(ct - self._start)
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]])
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)

    def _timestamp(self, name):
        return time.time()


def benchmark(trainer):
    # Benchmark to achieve the backward time per layer
    p = Profiling(trainer.net)
    # Warmup
    input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):
        data = trainer.data_iter()

        if trainer.dataset == 'an4':
            inputs, labels_cpu, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        else:
            inputs, labels_cpu = data
        if trainer.is_cuda:
            if trainer.dnn == 'lstm' :
                inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
            else:
                inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
        else:
            labels = labels_cpu

        if trainer.dnn == 'lstman4':
            out, output_sizes = trainer.net(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            loss = trainer.criterion(out, labels_cpu, output_sizes, target_sizes)
            torch.cuda.synchronize()
            loss = loss / inputs.size(0)  # average the loss by minibatch
        elif trainer.dnn == 'lstm' :
            hidden = trainer.net.init_hidden()
            hidden = lstmpy.repackage_hidden(hidden)
            #print(inputs.size(), hidden[0].size(), hidden[1].size())
            outputs, hidden = trainer.net(inputs, hidden)
            tt = torch.squeeze(labels.view(-1, trainer.net.batch_size * trainer.net.num_steps))
            loss = trainer.criterion(outputs.view(-1, trainer.net.vocab_size), tt)
            torch.cuda.synchronize()
        else:
            # forward + backward + optimize
            outputs = trainer.net(inputs)
            loss = trainer.criterion(outputs, labels)
            torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        if trainer.is_cuda:
            torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1]

