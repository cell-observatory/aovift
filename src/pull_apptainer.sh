#!/bin/bash
ARG1=${1:-CUDA_11_8}
export APPTAINER_TMPDIR=/clusterfs/nvme/thayer/apptainer_cache/tmp
export APPTAINER_CACHEDIR=/clusterfs/nvme/thayer/apptainer_cache/cache
echo "This script will pull the CUDA version passed as the first argument ($ARG1), from the available packages (e.g. CUDA_11_8, CUDA12_3, etc)"
echo "Using APPTAINER_TMPDIR=$APPTAINER_TMPDIR and APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
echo "https://github.com/cell-observatory/aovift/pkgs/container/aovift"
echo "apptainer pull --force --disable-cache ../main_$ARG1.sif oras://ghcr.io/cell-observatory/aovift:main_$ARG1_sif"
apptainer pull --force --disable-cache ../main_$ARG1.sif oras://ghcr.io/cell-observatory/aovift:main_$ARG1_sif