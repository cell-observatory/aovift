from operator import floordiv
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    num_modes = 15
    window_size = (64, 64, 64)   # z-y-x
    digital_rotations = 361

    kargs = dict(
        repo=repo,
        inputs=repo / f'examples/beads.tif',
        fish_inputs=repo / f'examples/fish.tif',
        embeddings_shape=(6, 64, 64, 1),
        digital_rotations=digital_rotations,
        rotations_shape=(digital_rotations, 6, 64, 64, 1),
        window_size=window_size,
        num_modes=num_modes,
        model=repo / f'pretrained_models/aovift-{num_modes}-YuMB-lambda510.h5',
        dm_calibration=repo/'calibration/aang/15_mode_calibration.csv',
        psf_type=repo/'lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        ideal_psf=repo/'examples/ideal_psf.tif',
        prediction_filename_pattern=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        prev=None,
        dm_state=None,
        wavelength=.510,
        dm_damping_scalar=1.0,
        lateral_voxel_size=.097,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        confidence_threshold=0.02,
        num_predictions=1,
        batch_size=64,
        plot=True,
        plot_rotations=True,
        ignore_modes=[0, 1, 2, 4],
        # extra `aggregate_predictions` flags
        majority_threshold=.5,
        min_percentile=20,
        max_percentile=80,
        aggregation_rule='mean',
        max_isoplanatic_clusters=5,
        ignore_tile=[],
        # extra `predict_rois` flags
        num_rois=5,
        min_intensity=200,
        minimum_distance=.5,
        min_psnr=5,
        # limit the number of cpu workers to hopefully avoid "cupy.cuda.memory.OutOfMemoryError: Out of memory"
        # during "emb = rotate_embeddings(...)"
        big_job_cpu_workers=3,

    )

    return kargs
