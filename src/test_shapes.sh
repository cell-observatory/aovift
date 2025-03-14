#!/bin/bash

xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
SHAPE=64
MAXAMP=1.
DATA='/clusterfs/nvme/thayer/dataset/embeddings/test/x150-y150-z600/'

declare -a models=(
'../models/new/embeddings/transformers/p32-p16-p8x2/'
)

for RES in 256 384
do
  for MODEL in "${models[@]}"
  do
    for NA in 1 .95 .9 .85 .8
    do
      for REF in 'single_point' \
      '2_points_10p_radius' '5_points_10p_radius' '10_points_10p_radius' '25_points_10p_radius' '50_points_10p_radius' \
      '2_points_20p_radius' '5_points_20p_radius' '10_points_20p_radius' '25_points_20p_radius' '50_points_20p_radius' \
      '2_points_30p_radius' '5_points_30p_radius' '10_points_30p_radius' '25_points_30p_radius' '50_points_30p_radius' \
      '2_points_50p_radius' '5_points_50p_radius' '10_points_50p_radius' '25_points_50p_radius' '50_points_50p_radius' \
      'line' 'sheet' 'cylinder' \
      'point_and_line' 'point_and_sheet' 'point_and_cylinder' \
      'several_points_and_line' 'several_points_and_sheet' 'several_points_and_cylinder'
      do
        #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
        #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
        python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
        --task "$MODEL --na $NA --reference ../data/shapes/$RES/$REF.tif evalsample" \
        --taskname $NA \
        --name $MODEL/shapes/$RES/$REF
      done
    done
  done
done
