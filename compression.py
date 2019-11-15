# -*- coding: utf-8 -*-
from __future__ import print_function
import settings
import torch
import numpy as np
import time
import math
import utils
from scipy import stats


class NoneCompressor():
    @staticmethod
    def compress(tensor, name=None):
        return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'topk'

    @staticmethod
    def clear():
        TopKCompressor.residuals = {}
        TopKCompressor.sparsities = []
        TopKCompressor.zero_conditions = {}
        TopKCompressor.values = {} 
        TopKCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0. 

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes, values

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data
            #selected_indexes = TopKCompressor.indexes[name][indexes_t]
            #residuals.data[selected_indexes] = 0.0 
            #logger.info('residuals after: %f', torch.norm(TopKCompressor.residuals[name].data))

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 

class TopKCompressor2(TopKCompressor):
    name = 'topk2' # without residuals

    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            TopKCompressor.residuals[name].data[indexes] = 0. 

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes, values


class GaussianCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'gaussion'

    @staticmethod
    def clear():
        GaussianCompressor.residuals = {}
        GaussianCompressor.sparsities = []
        GaussianCompressor.zero_conditions = {}
        GaussianCompressor.values = {} 
        GaussianCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            #one_indexes = abs_tensor > right_thres
            #indexes = one_indexes.nonzero().data.squeeze().view(-1)
            #indexes = indexes #[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0 
            GaussianCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = GaussianCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = GaussianCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[GaussianCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 

class GaussianCompressor2(GaussianCompressor):
    name = 'gaussion2' # without residuals

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 5:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            indexes = indexes 
            values = tensor.data[indexes] 
            GaussianCompressor.residuals[name].data = tensor.data + 0.0 
            GaussianCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values



class RandomKCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'randomk'
    counter = 0

    @staticmethod
    def clear():
        RandomKCompressor.residuals = {}
        RandomKCompressor.sparsities = []
        RandomKCompressor.zero_conditions = {}
        RandomKCompressor.values = {} 
        RandomKCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            #tensor.add_(RandomKCompressor.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            RandomKCompressor.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes] 
            RandomKCompressor.residuals[name].data = tensor.data + 0.0 
            RandomKCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = RandomKCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = RandomKCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[RandomKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class RandomKECCompressor(RandomKCompressor):
    name = 'randomkec'

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RandomKCompressor.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes] 
            RandomKCompressor.residuals[name].data = tensor.data + 0.0 
            RandomKCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values


class RandomKSameCompressor(RandomKCompressor):
    name = 'randomksame'
    counter = 0

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            #torch.manual_seed(RandomKSCompressor.counter)
            #if tensor.is_cuda:
            #    torch.cuda.manual_seed_all(RandomKSCompressor.counter)
            torch.manual_seed(RandomKSameCompressor.counter)
            RandomKSameCompressor.counter += 1
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes] 
            RandomKCompressor.residuals[name].data = tensor.data + 0.0 
            RandomKCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values


class RandomKSameECCompressor(RandomKCompressor):
    name = 'randomksameec'
    counter = 0

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tensor.add_(RandomKCompressor.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes] 
            RandomKCompressor.residuals[name].data = tensor.data + 0.0 
            RandomKCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

class DGCSamplingCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'dgcsampling'

    @staticmethod
    def clear():
        DGCSamplingCompressor.residuals = {}
        DGCSamplingCompressor.sparsities = []
        DGCSamplingCompressor.zero_conditions = {}
        DGCSamplingCompressor.values = {} 
        DGCSamplingCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in DGCSamplingCompressor.residuals:
                DGCSamplingCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(DGCSamplingCompressor.residuals[name].data)

            abs_values = torch.abs(tensor)

            # First step sampling
            perm = torch.randperm(numel, device=tensor.device)
            fk = int(numel * 0.01)
            sampled_indexes = perm[0:fk]
            sampled_values = abs_values[sampled_indexes]
            tmpvalues, tmpindexes = torch.topk(sampled_values, k=k)

            thres = tmpvalues[k-1]
            bool_indexes = abs_values > thres
            indexes = bool_indexes.nonzero().data.squeeze().view(-1)
            num_k = len(indexes)
            if num_k > 4*k/3:
                tmpvalues = abs_values[indexes] 
                values, tmpindexes = torch.topk(tmpvalues, k=k)
                indexes = indexes[tmpindexes]

            values = tensor.data[indexes] 
            DGCSamplingCompressor.residuals[name].data = tensor.data + 0.0 
            DGCSamplingCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = DGCSamplingCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = DGCSamplingCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[DGCSamplingCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class RedSyncCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'redsync'

    @staticmethod
    def clear():
        RedSyncCompressor.residuals = {}
        RedSyncCompressor.sparsities = []
        RedSyncCompressor.zero_conditions = {}
        RedSyncCompressor.values = {} 
        RedSyncCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RedSyncCompressor.residuals[name].data)

            l = 0.0
            r = 1.0
            thres = 0.0
            eps = 0.2
            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)

            while r - l > eps:
                tmp_ratio = l + (r-l)/2
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                nnz = indexes.numel()
                if nnz > k and 2*k > nnz:
                    break
                elif nnz < k/2:
                    r = tmp_ratio
                else:
                    l = tmp_ratio
            indexes = indexes 
            values = tensor.data[indexes] 
            RedSyncCompressor.residuals[name].data = tensor.data + 0.0 
            RedSyncCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = RedSyncCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = RedSyncCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[RedSyncCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class RedSyncTrimCompressor(RedSyncCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'redsynctrim'

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RedSyncCompressor.residuals[name].data)

            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)
            eps = 0.2
            tmp_ratio = 1 - eps

            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            nnz = indexes.numel()

            while nnz < k:
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                nnz = indexes.numel()
                tmp_ratio = tmp_ratio - eps
               
            indexes = indexes 
            values = tensor.data[indexes] 
            RedSyncCompressor.residuals[name].data = tensor.data + 0.0 
            RedSyncCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values


compressors = {
        'topk': TopKCompressor,
        'topk2': TopKCompressor2,
        'gaussian': GaussianCompressor,
        'gaussian2': GaussianCompressor2,
        'randomk': RandomKCompressor,
        'randomkec': RandomKECCompressor,
        'dgcsampling': DGCSamplingCompressor,
        'redsync': RedSyncCompressor,
        'redsynctrim': RedSyncTrimCompressor,
        'none': NoneCompressor,
        None: NoneCompressor
        }


def test_gaussion_thres():
    set_mean = 0.0; set_std = 0.5
    d = np.random.normal(set_mean, set_std, 10000)
    k2, p = stats.normaltest(d)
    print(p)
    nnz = np.count_nonzero(d)
    mean = np.mean(d)
    std = np.std(d)
    print('size:%d, nnz: %d' % (d.size, nnz))
    print(set_mean, set_std)
    print(mean, std)
    thres = 3*std
    d[np.abs(d) < thres] = 0
    pvalue = 1-np.count_nonzero(d)*1.0/d.size
    print('size:%d, p-value: %f' % (d.size, pvalue))
    left_thres, right_thres = utils.gen_threshold_from_normal_distribution(pvalue, mean, std)
    print('real thres:%f, gen thres: %f' % (thres, right_thres))


if __name__ == '__main__':
    test_gaussion_thres()
