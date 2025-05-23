#!/bin/bash

OUTDIR='../models/new/variable_object_size_fourier_filter_125nm_dataset/'
PRETRAINED="../models/new/variable_object_size_fourier_filter_125nm_dataset"
CLUSTER='slurm'
TIMELIMIT='24:00:00'  #hh:mm:ss
APPTAINER="--apptainer ../main_tf_cuda_12_3.sif"
JOB="benchmark.py --timelimit $TIMELIMIT --partition abc_a100 --mem=500GB --cpus 16 --gpus 1"

TRAINED_MODELS=(
  "vit-T32 vit/vit-15-YuMB_lambda510-T32"
  "vit-S32 vit/vit-15-YuMB_lambda510-S32"
  "vit-B32 vit/vit-15-YuMB_lambda510-B32"
  "vit-L32 vit/vit-15-YuMB_lambda510-L32"
  "vit-T16 vit/vit-15-YuMB_lambda510-T16"
  "vit-S16 vit/vit-15-YuMB_lambda510-S16"
  "vit-B16 vit/vit-15-YuMB_lambda510-B16"
  "convnext-T convnext/convnext-15-YuMB_lambda510-T"
  "convnext-S convnext/convnext-15-YuMB_lambda510-S"
  "convnext-B convnext/convnext-15-YuMB_lambda510-B"
  "convnext-L convnext/convnext-15-YuMB_lambda510-L"
  "aovift-T3216 scaling/aovift-15-YuMB_lambda510-T3216"
  "aovift-S3216 scaling/aovift-15-YuMB_lambda510-S3216"
  "aovift-B3216 scaling/aovift-15-YuMB_lambda510-B3216"
  "aovift-L3216 scaling/aovift-15-YuMB_lambda510-L3216"
  "aovift-H3216 scaling/aovift-15-YuMB_lambda510-H3216"
)

for S in `seq 1 ${#TRAINED_MODELS[@]}`
do
    set -- ${TRAINED_MODELS[$S-1]}
    M=$1
    P=$2

    echo Profile $M
    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "profile_models ../ --outdir ${OUTDIR} --model_codename ${M} --model_predictions ${PRETRAINED}/${P}" \
    --taskname profile_models \
    --name ${OUTDIR}/${M}

    echo '----------------'
    echo
done


TRAINED_MODELS=(
  "32-32-32-16 multistage/aovift-15-YuMB_lambda510-P32323216-R2222-H8888"
  "32-32-16-16 multistage/aovift-15-YuMB_lambda510-P32321616-R2222-H8888"
  "32-16-16-16 multistage/aovift-15-YuMB_lambda510-P32161616-R2222-H8888"
  "32-16-16-8 multistage/aovift-15-YuMB_lambda510-P3216168-R2222-H8888"
  "32-16-8-8 multistage/aovift-15-YuMB_lambda510-P321688-R2222-H8888"
)

for S in `seq 1 ${#TRAINED_MODELS[@]}`
do
    set -- ${TRAINED_MODELS[$S-1]}
    M=$1
    P=$2

    echo Profile $M
    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "profile_stages ../ --outdir ${OUTDIR} --model_codename ${M} --model_predictions ${PRETRAINED}/${P}" \
    --taskname profile_stages \
    --name ${OUTDIR}/${M}

    echo '----------------'
    echo
done
