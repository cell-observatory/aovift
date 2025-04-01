
AOViFT: Adaptive Optical Vision Fourier Transformer 
====================================================
[![arXiv](https://img.shields.io/badge/arXiv-2503.12593-b31b1b.svg)](https://arxiv.org/abs/2503.12593)
[![package](https://github.com/cell-observatory/aovift/actions/workflows/docker_action.yml/badge.svg)](https://github.com/cell-observatory/aovift/actions/workflows/docker_action.yml)
[![python](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![tensorflow](https://img.shields.io/badge/tensorFlow-2.14+-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![license](https://img.shields.io/github/license/cell-observatory/aovift.svg?style=flat&logo=git&logoColor=white)](https://opensource.org/license/bsd-2-clause/)
[![issues](https://img.shields.io/github/issues/cell-observatory/aovift.svg?style=flat&logo=github)](https://github.com/cell-observatory/aovift/issues)
[![pr](https://img.shields.io/github/issues-pr/cell-observatory/aovift.svg?style=flat&logo=github)](https://github.com/cell-observatory/aovift/pulls)


<div style="text-align: center; width: 100%; display: inline-block; text-align: center;" >
 <h2>Fourier-Based 3D Multistage Transformer for Aberration Correction in Multicellular Specimens</h2>
  <p>
  Thayer Alshaabi<sup>1,2*</sup>, Daniel E. Milkie<sup>1</sup>, Gaoxiang Liu<sup>2</sup>, Cyna Shirazinejad<sup>2</sup>, Jason L. Hong<sup>2</sup>, Kemal Achour<sup>2</sup>, Frederik Görlitz<sup>2</sup>, Ana Milunovic-Jevtic<sup>2</sup>, Cat Simmons<sup>2</sup>, Ibrahim S. Abuzahriyeh<sup>2</sup>, Erin Hong<sup>2</sup>, Samara Erin Williams<sup>2</sup>, Nathanael Harrison<sup>2</sup>, Evan Huang<sup>2</sup>, Eun Seok Bae<sup>2</sup>, Alison N. Killilea<sup>2</sup>, David G. Drubin<sup>2</sup>, Ian A. Swinburne<sup>2</sup>, Srigokul Upadhyayula<sup>2,3,4*</sup>, Eric Betzig<sup>1,2,5*</sup>
  </p>
  <h5>
    <sup>1</sup>HHMI, <sup>2</sup>UC Berkeley, <sup>3</sup>Lawrence Berkeley National Laboratory, <sup>4</sup>Chan Zuckerberg Biohub, <sup>5</sup>Helen Wills Neuroscience Institute
  </h5>
  <div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2503.12593-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.12593) &nbsp;
  [![Docker](https://img.shields.io/badge/docker-image-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/cell-observatory/aovift/pkgs/container/aovift) &nbsp;
  [![pretrained-models](https://img.shields.io/badge/pretrained-models-%233B4D98.svg?style=for-the-badge&logo=Dropbox&logoColor=white)](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&dl=0) &nbsp;
  [![Pytest](https://img.shields.io/badge/pytest-suite-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)](https://github.com/cell-observatory/aovift/tree/main/tests) &nbsp;
  [![BibTeX](https://img.shields.io/badge/BibTeX-reference-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)](#bibtex)

  </div>
</div>


<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/xonddbzyjptsh1c3me0y5/fish.gif?rlkey=7owu8ez9iuk1dyabbkj4idhrz&raw=1" width="100%" />
</div>

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/zc2b1qqd7wte2rxzw3qtg/model.png?rlkey=n7gtkbs6rq8jjk3mr9gwxc5zz&raw=1" width="100%" />
</div>

* [Overview](#overview)
* [System requirements](#system-requirements)
* [Installation](#installation)
* [Pretrained models](#pretrained-models)
* [Getting started](#getting-started)
  * [Synthetic data generator](#synthetic-data-generator)
  * [Fourier embedding](#fourier-embedding)
  * [Small FOV prediction](#small-fov-prediction)
  * [Tile-based prediction](#tile-based-prediction)
* [BibTeX](#bibtex)
* [License](#license)


# Overview

High-resolution tissue imaging is often compromised by sample-induced optical
aberrations that degrade resolution and contrast. While wavefront sensor-based
adaptive optics (AO) can measure these aberrations, such hardware solutions are
typically complex, expensive to implement, and slow when serially mapping
spatially varying aberrations across large fields of view. Here, we introduce
AOViFT (Adaptive Optical Vision Fourier Transformer)---a machine learning-based
aberration sensing framework built around a 3D multistage Vision Transformer
that operates on Fourier domain embeddings. AOViFT infers aberrations and
restores diffraction-limited performance in puncta-labeled specimens with
substantially reduced computational cost, training time, and memory footprint
compared to conventional architectures or real-space networks. We validated
AOViFT on live gene-edited zebrafish embryos, demonstrating its ability to
correct spatially varying aberrations using either a deformable mirror or
post-acquisition deconvolution. By eliminating the need for the guide star and
wavefront sensing hardware and simplifying the experimental workflow, AOViFT
lowers technical barriers for high-resolution volumetric microscopy across
diverse biological samples.

# System requirements

## OS requirements

Our source code is tested on the following operating systems:
- **Ubuntu 22.04** 
- **Rocky Linux 9.3**
- **Oracle Linux 9**
- **Windows 11 Pro** for Workstations (x64-based with 10.0.22621 build 22621) 

## Tested hardware

While our model can be used with similar hardware, 
we have tested our code on the following configurations:

**Microscope workstation:**
- Intel Xeon w5-3425, 12 cores
- [NVIDIA RTX A6000](https://resources.nvidia.com/en-us-briefcase-for-datasheets/proviz-print-nvidia-1?ncid=no-ncid) with 48GB of VRAM
- 512GB RAM

**A100 node:**
- 2 x AMD EPYC 7413, 24 cores
- 4 x [NVIDIA A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) with 80GB of VRAM
- 2TB RAM

**H100 node:**
- 2 x Intel Xeon Platinum 8468, 48 cores
- 8 x [NVIDIA H100](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) with 80GB of VRAM
- 4TB RAM 

## Driver requirements
Our docker image is based on the [NVIDIA Tensorflow docker image](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-02.html
) for the **24.02** release, 
which requires an NVIDIA GPU with a driver release **545** or later, and **CUDA 12.3.2**.

- NVIDIA CUDA 12.3.2
- cuDNN 9.0.0.306
- NCCL 2.19.4
- TensorRT 8.6.3 
- Python 3.10.6


# Installation

## Docker & Apptainer [images](https://github.com/cell-observatory/aovift/pkgs/container/aovift)
Our prebuilt image with Python, TensorFlow, and all packages installed for you.
```shell
docker pull ghcr.io/cell-observatory/aovift:main_tf_cuda_12_3
```

Running an image on a cluster typically requires an Apptainer version of the image, which can be generated by:
```shell
apptainer pull --force main_tf_cuda_12_3.sif docker://ghcr.io/cell-observatory/aovift:main_tf_cuda_12_3
```

## Clone repository to your host system
```shell
git clone --recurse-submodules https://github.com/cell-observatory/aovift.git
```

Run `tests/test_tensorflow.py` to make sure the installation works
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_tensorflow.py
```

**Note:** If you want to run a local version of the image, see the [Dockerfile](https://github.com/cell-observatory/aovift/blob/main/Dockerfile)

To later update to the latest, greatest
```shell
git pull --recurse-submodules
```


# Pretrained [models](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&dl=0)

All pre-trained models can be downloaded from our [pretrained models repository](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&dl=0).

If you wish to download all of our models at once, 
you can use this [link](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&raw=1) and extract the desired *.h5 file from the zip file.

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/5psg2uunus1xesa8doz28/benchmark.png?rlkey=iq2gbmnpn6idm1pc2k5fmmq6x&raw=1" width="100%" />
</div>

# Getting started

**Note:** 
To run AOViFT pytests you need to download the following files:
-  [Examples](https://www.dropbox.com/scl/fo/usvuuit2wsy7ycfj09g3g/AJShjIGk47KM14A4urHvteE?rlkey=ejj5hlfyur9sb0k6klshwwzdz&raw=1) directory that has a few example tif files and extract it to `aovift/examples`.
-  [aovift-15-YuMB-lambda510.h5](https://www.dropbox.com/scl/fi/tkkd50u5m40voy30g760r/aovift-15-YuMB-lambda510.h5?rlkey=gixl8i211kjlw092jgrk7mjm1&raw=1) and save it to `aovift/pretrained_models` directory.

The [`src/ao.py`](src/ao.py) script provides a CLI
for running our models on a given 3D stack (`.tif` file).


## Running docker image

To run docker image, replace `working-dir` with your local path for the repository.
```shell
docker run --network host -u 1000 --privileged -v working-dir/aovift:/app/aovift --env PYTHONUNBUFFERED=1 --pull missing -t -i --rm -w /app/aovift --ipc host --gpus all ghcr.io/cell-observatory/aovift:main_tf_cuda_12_3 bash
```

## [Synthetic data generator](https://github.com/cell-observatory/beads_simulator)

Running `tests/test_datasets.py` will create a dataset of synthetic data for testing.
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py
```

**Note:**
If you just want to use our beads simulator without our models for your own projects, you can use our [beads simulator repository](https://github.com/cell-observatory/beads_simulator).

## Fourier embedding

Running `tests/test_embeddings.py` will create a Fourier embeddings for testing.
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_embeddings.py
```
```shell
tests/test_embeddings.py::test_fourier_embeddings Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_embeddings.py::test_interpolate_embeddings Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_embeddings.py::test_rolling_fourier_embeddings Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Preprocessing, 1 rois per tile, roi size, (64, 64, 64), stride length [64 64 64], throwing away [0 0 0] voxels: 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  7.84it/s] 0.1s elapsed
Compute FFTs: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 50.52it/s] 0.0s elapsed
Remove interference patterns: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.04s/it] 1.0s elapsed
PASSED
tests/test_embeddings.py::test_embeddings_with_digital_rotations Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Preprocessing, 1 rois per tile, roi size, (64, 64, 64), stride length [64 64 64], throwing away [0 0 0] voxels: 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  7.85it/s] 0.1s elapsed
Compute FFTs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 149.57it/s] 0.0s elapsed
Remove interference patterns: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.05s/it] 1.0s elapsed
PASSED
```


## Small FOV prediction

<div align="center">
    <img src="https://www.dropbox.com/scl/fi/e1f0kpnoofa10moi85zvv/ap2.gif?rlkey=3pvphchl69dxgk5k72njt8brc&raw=1" width="100%" />
</div>

To predict the wavefront for a small FOV, you can use the `predict_sample` function: 

```shell
/app/aovift$ pytest -s -v --disable-pytest-warnings --color=yes tests/test_ao.py -k test_predict_sample
```
```shell
tests/test_ao.py::test_predict_sample Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
6/6 [==============================] - 8s 891ms/step
Evaluate predictions: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.26s/it] 3.3s elapsed
PASSED
```

```shell
usage: ao.py predict_sample [-h] [--current_dm CURRENT_DM] [--prev PREV] [--lateral_voxel_size LATERAL_VOXEL_SIZE] [--axial_voxel_size AXIAL_VOXEL_SIZE] [--wavelength WAVELENGTH]
                            [--dm_damping_scalar DM_DAMPING_SCALAR] [--freq_strength_threshold FREQ_STRENGTH_THRESHOLD] [--prediction_threshold PREDICTION_THRESHOLD]
                            [--confidence_threshold CONFIDENCE_THRESHOLD] [--sign_threshold SIGN_THRESHOLD] [--plot] [--plot_rotations] [--num_predictions NUM_PREDICTIONS] [--batch_size BATCH_SIZE]
                            [--estimate_sign_with_decon] [--ignore_mode IGNORE_MODE] [--ideal_empirical_psf IDEAL_EMPIRICAL_PSF] [--cpu_workers CPU_WORKERS] [--cluster] [--partition PARTITION] [--docker]
                            [--digital_rotations DIGITAL_ROTATIONS] [--psf_type PSF_TYPE] [--min_psnr MIN_PSNR] [--estimated_object_gaussian_sigma ESTIMATED_OBJECT_GAUSSIAN_SIGMA] [--denoiser DENOISER]
                            model input dm_calibration

positional arguments:
  model                 path to pretrained tensorflow model
  input                 path to input .tif file
  dm_calibration        path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)

options:
  -h, --help            show this help message and exit
  --current_dm CURRENT_DM
                        optional path to current DM .csv file (Default: `blank mirror`)
  --prev PREV           previous predictions .csv file (Default: `None`)
  --lateral_voxel_size LATERAL_VOXEL_SIZE
                        lateral voxel size in microns for X (Default: `0.097`)
  --axial_voxel_size AXIAL_VOXEL_SIZE
                        axial voxel size in microns for Z (Default: `0.1`)
  --wavelength WAVELENGTH
                        wavelength in microns (Default: `0.51`)
  --dm_damping_scalar DM_DAMPING_SCALAR
                        scale DM actuators by an arbitrary multiplier (Default: `0.75`)
  --freq_strength_threshold FREQ_STRENGTH_THRESHOLD
                        minimum frequency threshold in fourier space (percentages; values below that will be set to the desired minimum) (Default: `0.01`)
  --prediction_threshold PREDICTION_THRESHOLD
                        set predictions below threshold to zero (waves) (Default: `0.0`)
  --confidence_threshold CONFIDENCE_THRESHOLD
                        optional threshold to flag unconfident predictions based on the standard deviations of the predicted amplitudes for all digital rotations (microns) (Default: `0.02`)
  --sign_threshold SIGN_THRESHOLD
                        flip sign of modes above given threshold relative to your initial prediction (Default: `0.9`)
  --plot                a toggle for plotting predictions
  --plot_rotations      a toggle for plotting predictions for digital rotations
  --num_predictions NUM_PREDICTIONS
                        number of predictions per sample to estimate model's confidence (Default: `1`)
  --batch_size BATCH_SIZE
                        maximum batch size for the model (Default: `100`)
  --estimate_sign_with_decon
                        a toggle for estimating signs of each Zernike mode via decon
  --ignore_mode IGNORE_MODE
                        ANSI index for mode you wish to ignore (Default: `[0, 1, 2, 4]`)
  --ideal_empirical_psf IDEAL_EMPIRICAL_PSF
                        path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)
  --cpu_workers CPU_WORKERS
                        number of CPU cores to use (Default: `-1`)
  --cluster             a toggle to run predictions on our cluster
  --partition PARTITION
                        slurm partition to use on the ABC cluster (Default: `abc_a100`)
  --docker              a toggle to run predictions through docker container
  --digital_rotations DIGITAL_ROTATIONS
                        optional flag for applying digital rotations (Default: `361`)
  --psf_type PSF_TYPE   widefield, 2photon, confocal, or a path to an LLS excitation profile (Default: None; to keep default mode used during training)
  --min_psnr MIN_PSNR   Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold (Default: `5`)
  --estimated_object_gaussian_sigma ESTIMATED_OBJECT_GAUSSIAN_SIGMA
                        size of object for creating an ideal psf (default: 0; single pixel) (Default: `0.0`)
  --denoiser DENOISER   path to denoiser model (Default: `None`)
```


## Tile-based prediction

<div align="center">
  <img src="https://www.dropbox.com/scl/fi/jkfvlahfsgnrgxskmcjw0/fishmap.png?rlkey=cbs6fy4o23zm23v3ay88lexa1&raw=1" width="100%" />
</div>

To tile a large FOV and predict the wavefront of each tile, you can use the `predict_tiles` function:  
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_ao.py -k test_predict_tiles
```
```shell
tests/test_ao.py::test_predict_tiles Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.125_y_0.125_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Locating tiles: [288]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288 [00:41<00:00,  6.90 tile/s] 41.7s elapsed
Generate fourier embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [00:40<00:00,  4.17 .tif file/s] 41.0s elapsed
965/965 [==============================] - 950s 976ms/step
Evaluate predictions: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [00:00<00:00, 857925.82 evals/s] 0.0s elapsed
PASSED
```

You can then run `aggregate_predictions` to create some visualizations : 
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_ao.py -k test_aggregate_tiles
```
```shell
tests/test_ao.py::test_aggregate_tiles Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /app/aovift/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.59_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False

Number of tiles in each cluster of aggregated map, z=0
c    count
1     8
2    23
3    12
4    14
5    30
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288 [00:08<00:00, 34.95it/s]
PASSED
```

```shell
usage: ao.py predict_tiles [-h] [--current_dm CURRENT_DM] [--batch_size BATCH_SIZE] [--window_size WINDOW_SIZE] [--prev PREV] [--lateral_voxel_size LATERAL_VOXEL_SIZE]
                           [--axial_voxel_size AXIAL_VOXEL_SIZE] [--wavelength WAVELENGTH] [--freq_strength_threshold FREQ_STRENGTH_THRESHOLD] [--confidence_threshold CONFIDENCE_THRESHOLD]
                           [--sign_threshold SIGN_THRESHOLD] [--estimated_object_gaussian_sigma ESTIMATED_OBJECT_GAUSSIAN_SIGMA] [--plot] [--plot_rotations] [--num_predictions NUM_PREDICTIONS]
                           [--estimate_sign_with_decon] [--ignore_mode IGNORE_MODE] [--ideal_empirical_psf IDEAL_EMPIRICAL_PSF] [--cpu_workers CPU_WORKERS] [--cluster] [--partition PARTITION] [--docker]
                           [--digital_rotations DIGITAL_ROTATIONS] [--shift SHIFT] [--psf_type PSF_TYPE] [--min_psnr MIN_PSNR] [--denoiser DENOISER]
                           model input dm_calibration

positional arguments:
  model                 path to pretrained tensorflow model
  input                 path to input .tif file
  dm_calibration        path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)

options:
  -h, --help            show this help message and exit
  --current_dm CURRENT_DM
                        optional path to current DM .csv file (Default: `blank mirror`)
  --batch_size BATCH_SIZE
                        maximum batch size for the model (Default: `100`)
  --window_size WINDOW_SIZE
                        size of the window to crop each tile (Default: `64-64-64`)
  --prev PREV           previous predictions .csv file (Default: `None`)
  --lateral_voxel_size LATERAL_VOXEL_SIZE
                        lateral voxel size in microns for X (Default: `0.097`)
  --axial_voxel_size AXIAL_VOXEL_SIZE
                        axial voxel size in microns for Z (Default: `0.1`)
  --wavelength WAVELENGTH
                        wavelength in microns (Default: `0.51`)
  --freq_strength_threshold FREQ_STRENGTH_THRESHOLD
                        minimum frequency threshold in fourier space (percentages; values below that will be set to the desired minimum) (Default: `0.01`)
  --confidence_threshold CONFIDENCE_THRESHOLD
                        optional threshold to flag unconfident predictions based on the standard deviations of the predicted amplitudes for all digital rotations (microns) (Default: `0.015`)
  --sign_threshold SIGN_THRESHOLD
                        flip sign of modes above given threshold relative to your initial prediction (Default: `0.9`)
  --estimated_object_gaussian_sigma ESTIMATED_OBJECT_GAUSSIAN_SIGMA
                        size of object for creating an ideal psf (default: 0; single pixel) (Default: `0.0`)
  --plot                a toggle for plotting predictions
  --plot_rotations      a toggle for plotting predictions for digital rotations
  --num_predictions NUM_PREDICTIONS
                        number of predictions per tile to estimate model's confidence (Default: `1`)
  --estimate_sign_with_decon
                        a toggle for estimating signs of each Zernike mode via decon
  --ignore_mode IGNORE_MODE
                        ANSI index for mode you wish to ignore (Default: `[0, 1, 2, 4]`)
  --ideal_empirical_psf IDEAL_EMPIRICAL_PSF
                        path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)
  --cpu_workers CPU_WORKERS
                        number of CPU cores to use (Default: `-1`)
  --cluster             a toggle to run predictions on our cluster
  --partition PARTITION
                        slurm partition to use on the ABC cluster (Default: `abc_a100`)
  --docker              a toggle to run predictions through docker container
  --digital_rotations DIGITAL_ROTATIONS
                        optional flag for applying digital rotations (Default: `361`)
  --shift SHIFT         optional flag for applying digital x shift (Default: `0`)
  --psf_type PSF_TYPE   widefield, 2photon, confocal, or a path to an LLS excitation profile (Default: None; to keep default mode used during training)
  --min_psnr MIN_PSNR   Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold (Default: `5`)
  --denoiser DENOISER   path to denoiser model (Default: `None`)
```

```shell
usage: ao.py aggregate_predictions [-h] [--current_dm CURRENT_DM] [--dm_damping_scalar DM_DAMPING_SCALAR] [--prediction_threshold PREDICTION_THRESHOLD] [--majority_threshold MAJORITY_THRESHOLD]
                                   [--aggregation_rule AGGREGATION_RULE] [--min_percentile MIN_PERCENTILE] [--max_percentile MAX_PERCENTILE] [--max_isoplanatic_clusters MAX_ISOPLANATIC_CLUSTERS] [--plot]
                                   [--ignore_tile IGNORE_TILE] [--cpu_workers CPU_WORKERS] [--cluster] [--partition PARTITION] [--docker] [--psf_type PSF_TYPE]
                                   input dm_calibration

positional arguments:
  input                 path to csv file
  dm_calibration        path DM calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)

options:
  -h, --help            show this help message and exit
  --current_dm CURRENT_DM
                        optional path to current DM current_dm .csv file (Default: `blank mirror`)
  --dm_damping_scalar DM_DAMPING_SCALAR
                        scale DM actuators by an arbitrary multiplier (Default: `0.75`)
  --prediction_threshold PREDICTION_THRESHOLD
                        set predictions below threshold to zero (p2v waves) (Default: `0.25`)
  --majority_threshold MAJORITY_THRESHOLD
                        majority rule to use to determine dominant modes among ROIs (Default: `0.5`)
  --aggregation_rule AGGREGATION_RULE
                        rule to use to calculate final prediction [mean, median, min, max] (Default: `median`)
  --min_percentile MIN_PERCENTILE
                        minimum percentile to filter out outliers (Default: `5`)
  --max_percentile MAX_PERCENTILE
                        maximum percentile to filter out outliers (Default: `95`)
  --max_isoplanatic_clusters MAX_ISOPLANATIC_CLUSTERS
                        maximum number of unique isoplanatic patchs for clustering tiles (Default: `3`)
  --plot                a toggle for plotting predictions
  --ignore_tile IGNORE_TILE
                        IDs [e.g., "z0-y0-x0"] for tiles you wish to ignore
  --cpu_workers CPU_WORKERS
                        number of CPU cores to use (Default: `-1`)
  --cluster             a toggle to run predictions on our cluster
  --partition PARTITION
                        slurm partition to use on the ABC cluster (Default: `abc_a100`)
  --docker              a toggle to run predictions through docker container
  --psf_type PSF_TYPE   widefield, 2photon, confocal, or a path to an LLS excitation profile (Default: None; to keep default mode used during training)
```


# BibTeX

```bibtex
@article{alshaabi2025fourier,
  title={Fourier-Based 3D Multistage Transformer for Aberration Correction in Multicellular Specimens},
  author={Thayer Alshaabi and Daniel E. Milkie and Gaoxiang Liu and Cyna Shirazinejad and Jason L. Hong and Kemal Achour and Frederik Görlitz and Ana Milunovic-Jevtic and Cat Simmons and Ibrahim S. Abuzahriyeh and Erin Hong and Samara Erin Williams and Nathanael Harrison and Evan Huang and Eun Seok Bae and Alison N. Killilea and David G. Drubin and Ian A. Swinburne and Srigokul Upadhyayula and Eric Betzig},
  journal={arXiv preprint arXiv:2503.12593},
  year={2025},
  url={https://arxiv.org/abs/2503.12593},
}
```

# License 

This work is licensed under the [BSD 2-Clause License](https://github.com/cell-observatory/aovift/blob/main/LICENSE)
