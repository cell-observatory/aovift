
import logging
import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import warnings
warnings.filterwarnings("ignore")

import pytest
from pathlib import Path
try:
    import cupy as cp
    use_gpu = True
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except ImportError as e:
    use_gpu = False
    logging.warning(f"Cupy not supported on your system: {e}")

from src import experimental


def cleanup_mempool():
    if use_gpu:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    else:
        pass

@pytest.mark.run(order=1)
def test_predict_tiles(kargs):
    logging.info(
        f"Pytest will assert that 'tile_predictions' has output shape of: "
        f"(num_modes={kargs['num_modes']}, "
        f"since window_size={kargs['window_size']}."
    )
    tile_predictions = experimental.predict_tiles(
        model=kargs['model'],
        img=kargs['fish_inputs'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=False,
        plot_rotations=False,
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        window_size=kargs['window_size'],
        min_psnr=kargs['min_psnr']
    )
    tile_predictions = tile_predictions.drop(columns=['mean', 'median', 'min', 'max', 'std'])
    assert tile_predictions.shape[0] == kargs['num_modes']


@pytest.mark.run(order=2)
def test_aggregate_tiles(kargs):
    zernikes = experimental.aggregate_predictions(
        model_pred=Path(f"{kargs['fish_inputs'].with_suffix('')}_tiles_predictions.csv"),
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prediction_threshold=kargs['prediction_threshold'],
        majority_threshold=kargs['majority_threshold'],
        min_percentile=kargs['min_percentile'],
        max_percentile=kargs['max_percentile'],
        aggregation_rule=kargs['aggregation_rule'],
        max_isoplanatic_clusters=kargs['max_isoplanatic_clusters'],
        ignore_tile=kargs['ignore_tile'],
        plot=kargs['plot'],
    )
    assert zernikes.shape[1] == kargs['num_modes'] + 3


@pytest.mark.run(order=3)
def test_predict_sample(kargs):
    zernikes = experimental.predict_sample(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr']
    )
    assert zernikes.shape[0] == kargs['num_modes']

@pytest.mark.run(order=4)
def test_phase_retrieval(kargs):
    zernikes = experimental.phase_retrieval(
        img=kargs['inputs'],
        num_modes=kargs['num_modes'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        ignore_modes=kargs['ignore_modes'],
        plot=kargs['plot'],
        prediction_threshold=kargs['prediction_threshold'],
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=5)
def test_predict_large_fov(kargs):
    zernikes = experimental.predict_large_fov(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr']
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=6)
def test_predict_large_fov_with_interpolated_embeddings(kargs):
    cleanup_mempool()
    zernikes = experimental.predict_large_fov(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr'],
        interpolate_embeddings=True
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=7)
def test_predict_folder(kargs):
    cleanup_mempool()
    test_folder = Path(f"{kargs['repo']}/dataset/experimental_zernikes/psfs")
    number_of_files = len(sorted(test_folder.glob(kargs['prediction_filename_pattern'])))
    
    logging.info(
        f"Pytest will assert that 'folder_predictions' has output shape of: "
        f"(num_modes={kargs['num_modes']}, number_of_files={number_of_files})"
    )
    
    predictions = experimental.predict_folder(
        model=kargs['model'],
        folder=test_folder,
        filename_pattern=kargs['prediction_filename_pattern'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        min_psnr=kargs['min_psnr']
    )
    predictions = predictions.drop(columns=['mean', 'median', 'min', 'max', 'std'])
    assert predictions.shape == (kargs['num_modes'], number_of_files)

