#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-0.001}"
compressor="${compressor:-topk}"
nwpernode=4
nstepsupdate=1
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
GRADSPATH=./logs/iclr

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_DEBUG=INFO \
    $PY dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --saved-dir $GRADSPATH 
