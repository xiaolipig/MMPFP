#!/bin/bash

echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM] [RANK_SIZE]"

ROOT_PATH=`pwd`
export RANK_TABLE_FILE=$1
export SERVICE_ID=0
export DEVICE_NUM=$2
export RANK_SIZE=$3
export SPLIT_SIZE_X=6000
export SPLIT_SIZE_Y=6000
rank_start=$((DEVICE_NUM * SERVICE_ID))

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    rm ${ROOT_PATH}/device_train$i/ -rf
    mkdir ${ROOT_PATH}/device_train$i
    cd ${ROOT_PATH}/device_train$i || exit
    mpirun -n 3 --allow-run-as-root python ${ROOT_PATH}/ssseg/luojianet_train_new.py > log$i.log 2>&1 &
done
