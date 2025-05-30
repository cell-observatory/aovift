import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import os, platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

import logging
import sys
import h5py

from pathlib import Path
from typing import Any, Union, Optional, Callable
from itertools import repeat
from functools import partial
from line_profiler_pycharm import profile
from tifffile import imwrite

import pandas as pd
from skimage.restoration import richardson_lucy
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import scipy as sp
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow_addons.optimizers import AdamW, LAMB  # keep for old models trained with TF2.5
from csbdeep.models import CARE

import psutil
import multiprocessing as mp
if platform.system() != "Windows":
    from dask_cuda import LocalCUDACluster  # Only Linux is supported by Dask-CUDA at this time

from dask.distributed import LocalCluster, Client

mp.set_start_method('spawn', force=True)

import utils
import vis
import data_utils

from synthetic import SyntheticPSF
from wavefront import Wavefront
from embeddings import fourier_embeddings, rolling_fourier_embeddings
from preprocessing import prep_sample
from utils import round_to_even, multiprocess

from roi import ROI
from activation import MaskedActivation
from depthwiseconv import DepthwiseConv3D
from spatial import SpatialAttention
from stochasticdepth import StochasticDepth
from stem import Stem

import convnext
import aovift
import prototype
import vit
from warmupcosinedecay import WarmupCosineDecay


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


class DatasetGenerator:
    def __init__(self, paths: np.ndarray, load_preprocess_func: Callable):
        self.paths = paths
        self.load_preprocess_func = load_preprocess_func
        self.gpus = len(tf.config.list_physical_devices('GPU'))

        if self.gpus >= 1 and platform.system() != 'Windows':
            self.client = Client(LocalCUDACluster(
                CUDA_VISIBLE_DEVICES=np.arange(self.gpus),
                threads_per_worker=1
            ))
        else:
            self.client = Client(LocalCluster(
                n_workers=psutil.cpu_count(logical=False)
            ))

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        x = self.paths[idx]
        x = self.client.submit(self.load_preprocess_func, x)
        x = x.result()
        return x

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


@profile
def load(model_path: Union[Path, str], mosaic=False, model_arch=None) -> tf.keras.Model:
    model_path = Path(model_path)

    if mosaic:
        if model_path.is_file() and model_path.suffix == '.h5':
            model_path = Path(model_path)
        else:
            model_path = Path(sorted(model_path.rglob('*.h5'), reverse=True)[0])
    else:
        try:
            '''.pb format'''
            if model_path.is_file() and model_path.suffix == '.pb':
                model_path = str(model_path.parent)
            else:
                model_path = str(sorted(model_path.rglob('saved_model.pb'), reverse=True)[0].parent)

        except IndexError or FileNotFoundError or OSError:
            '''.h5/hdf5 format'''
            if model_path.is_file() and model_path.suffix == '.h5':
                model_path = str(model_path)
            else:
                model_path = str(sorted(model_path.rglob('*.h5'), reverse=True)[0])

    logger.info(f"Loading {model_path}")
    custom_objects = {
        "LAMB": LAMB,
        "AdamW": AdamW,
        "WarmupCosineDecay": WarmupCosineDecay,
        "ROI": ROI,
        "MaskedActivation": MaskedActivation,
        "SpatialAttention": SpatialAttention,
        "DepthwiseConv3D": DepthwiseConv3D,
        "StochasticDepth": StochasticDepth,
        "Stem": Stem,
    }
    
    aovift_custom_objects = {
        "LAMB": LAMB,
        "AdamW": AdamW,
        "WarmupCosineDecay": WarmupCosineDecay,
        "ROI": ROI,
        "Stem": aovift.Stem,
        "Patchify": aovift.Patchify,
        "Merge": aovift.Merge,
        "PatchEncoder": aovift.PatchEncoder,
        "MLP": aovift.MLP,
        "Transformer": aovift.Transformer,
    }
    
    prototype_custom_objects = {
        "LAMB": LAMB,
        "AdamW": AdamW,
        "WarmupCosineDecay": WarmupCosineDecay,
        "ROI": ROI,
        "Patchify": prototype.Patchify,
        "PatchEncoder": prototype.PatchEncoder,
        "MLP": prototype.MLP,
        "Transformer": prototype.Transformer,
    }
    
    vit_custom_objects = {
        "LAMB": LAMB,
        "AdamW": AdamW,
        "WarmupCosineDecay": WarmupCosineDecay,
        "ROI": ROI,
        "Patchify": vit.Patchify,
        "PatchEncoder": vit.PatchEncoder,
        "MLP": vit.MLP,
        "Transformer": vit.Transformer,
    }
    
    convnext_custom_objects = {
        "LAMB": LAMB,
        "GRN": convnext.GRN,
        "StochasticDepth": convnext.StochasticDepth,
        "Block": convnext.Block,
        "AdamW": AdamW,
        "WarmupCosineDecay": WarmupCosineDecay,
        "ROI": ROI,
    }
    
    if model_arch is None:
        
        try:
            return load_model(model_path, custom_objects=aovift_custom_objects)
        except TypeError as e:
            return load_model(model_path, custom_objects=prototype_custom_objects)
        
    elif model_arch.lower() == 'aovift':
        return load_model(model_path, custom_objects=aovift_custom_objects)
    
    elif model_arch.lower() == 'prototype':
        return load_model(model_path, custom_objects=prototype_custom_objects)
    
    elif model_arch.lower() == 'vit':
        return load_model(model_path, custom_objects=vit_custom_objects)
    
    elif model_arch.lower() == 'convnext':
        return load_model(model_path, custom_objects=convnext_custom_objects)

    else:
        return load_model(model_path, custom_objects=custom_objects)
        


@profile
def load_metadata(
        model_path: Path,
        psf_shape: Union[tuple, list] = (64, 64, 64),
        psf_type=None,
        n_modes=None,
        x_voxel_size=None,
        y_voxel_size=None,
        z_voxel_size=None,
        lam_detection=None,
        **kwargs
):
    """ The model .h5, HDF5 file, is also used to store metadata parameters (wavelength, x_voxel_size, etc) that
    the model was trained with.  The metadata is read from file and is returned within the returned SyntheticPSF class.

    Args:
        model_path (Path): path to .h5 model, or path to the containing folder.
        psf_shape (Union[tuple, list], optional): dimensions of the SyntheticPSF to return. Defaults to (64, 64, 64).
        psf_type (str, optional): codename or path for the PSF type eg. "widefield". Defaults to None which reads the value from the .h5 file.
        n_modes (int, optional): # of Zernike modes to describe aberration. Defaults to None which reads the value from the .h5 file.
        z_voxel_size (float, optional):  Defaults to None which reads the value from the .h5 file.
        **kwargs:  Get passed into SyntheticPSF generator.

    Returns:
        SyntheticPSF class: ideal PSF that the model was trained on.
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    # print(f"my suffix = {model_path.suffix}, my model = {model_path}")
    if not model_path.suffix == '.h5':
        model_path = list(model_path.rglob('*.h5'))[0]  # locate the model if the parent folder path is given

    with h5py.File(model_path, 'r') as file:

        try:
            embedding_option = str(file.get('embedding_option')[()]).strip("b'").strip("\'").strip('\"')
        except Exception:
            embedding_option = 'spatial_planes'

        psfgen = SyntheticPSF(
            psf_type=str(file.get('psf_type')[()]).strip("b'").strip("\'").strip('\"') if psf_type is None else psf_type,
            psf_shape=psf_shape,
            n_modes=int(file.get('n_modes')[()]) if n_modes is None else n_modes,
            lam_detection=float(file.get('wavelength')[()]) if lam_detection is None else lam_detection,
            x_voxel_size=float(file.get('x_voxel_size')[()]) if x_voxel_size is None else x_voxel_size,
            y_voxel_size=float(file.get('y_voxel_size')[()]) if y_voxel_size is None else y_voxel_size,
            z_voxel_size=float(file.get('z_voxel_size')[()]) if z_voxel_size is None else z_voxel_size,
            embedding_option=embedding_option,
            **kwargs
        )
    return psfgen


def save_metadata(
        filepath: Path,
        wavelength: float,
        psf_type: str,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        n_modes: int,
        embedding_option: str = 'spatial_planes',
        skip_remove_background: bool = False,
        use_theoretical_widefield_simulator: bool = False,
):
    def add_param(h5file, name, data):
        try:
            if name not in h5file.keys():
                h5file.create_dataset(name, data=data)
            else:
                del h5file[name]
                h5file.create_dataset(name, data=data)

            if isinstance(data, str):
                if h5file.get(name)[()] == data:
                    logger.info(f"`{name}` : {data}")
                elif h5file.get(name)[()] == bytes(data, 'ASCII'):
                    logger.info(f"`{name}` : {data}")
                else:
                    logger.error(f"{name} has value of {h5file.get(name)[()]}, but we wanted '{data}'")
            else:
                assert np.allclose(h5file.get(name)[()], data), f"Failed to write {name}"
                logger.info(f"`{name}`: {h5file.get(name)[()]}")

        except Exception as e:
            logger.error(e)

    try:
        file = filepath if str(filepath).endswith('.h5') else list(filepath.rglob('*.h5'))[0]
    except IndexError:
        model = load(filepath)
        file = Path(f"{filepath}.h5")
        save_model(model, file, save_format='h5')

    with h5py.File(file, 'r+') as file:
        add_param(file, name='n_modes', data=n_modes)
        add_param(file, name='wavelength', data=wavelength)
        add_param(file, name='x_voxel_size', data=x_voxel_size)
        add_param(file, name='y_voxel_size', data=y_voxel_size)
        add_param(file, name='z_voxel_size', data=z_voxel_size)
        add_param(file, name='embedding_option', data=embedding_option)
        add_param(file, name='skip_remove_background', data=skip_remove_background)
        add_param(file, name='use_theoretical_widefield_simulator', data=use_theoretical_widefield_simulator)

        if (isinstance(psf_type, Path) or isinstance(psf_type, str)) and Path(psf_type).exists():
            with h5py.File(psf_type, 'r+') as f:
                add_param(file, name='psf_type', data=psf_type)
                add_param(file, name='lls_excitation_profile', data=f.get('DitheredxzPSFCrossSection')[:, 0])
        else:
            add_param(file, name='psf_type', data=psf_type)
            add_param(file, name='lls_excitation_profile', data=[])

    logger.info(f"Saved model with additional metadata: {filepath.resolve()}")


@profile
def load_sample(data: Union[tf.Tensor, Path, str, np.ndarray]):
    if isinstance(data, np.ndarray):
        img = data.astype(np.float32)

    elif isinstance(data, bytes):
        data = Path(str(data, "utf-8"))
        img = data_utils.get_image(data)

    elif isinstance(data, tf.Tensor):
        path = Path(str(data.numpy(), "utf-8"))
        img = data_utils.get_image(path)

    else:
        path = Path(str(data))
        img = data_utils.get_image(path)

    return np.squeeze(img)


def preprocess(
    file: Union[tf.Tensor, Path, str],
    modelpsfgen: SyntheticPSF,
    samplepsfgen: Optional[SyntheticPSF] = None,
    freq_strength_threshold: float = .01,
    digital_rotations: Optional[int] = 361,
    remove_background: bool = True,
    read_noise_bias: float = 5,
    normalize: bool = True,
    plot: Any = None,
    no_phase: bool = False,
    fov_is_small: bool = True,
    rolling_strides: Optional[tuple] = None,
    skip_prep_sample: bool = False,
    min_psnr: int = 10,
    cpu_workers: int = -1,
    denoiser: Optional[Union[Path, CARE]] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    interpolate_embeddings: bool = False,
    save_processed_tif_file: bool = False,
    estimated_object_gaussian_sigma: float = 0,
    use_reconstructed_otf: bool = False
):
    if samplepsfgen is None:
        samplepsfgen = modelpsfgen

    if isinstance(file, tf.Tensor):
        file = Path(str(file.numpy(), "utf-8"))

    if isinstance(plot, bool) and plot:
        plot = file.with_suffix('')

    sample = load_sample(file)
    
    # if estimated_object_gaussian_sigma != 0:
    #     kernel = utils.gaussian_kernel(kernlen=(21, 21, 21), std=estimated_object_gaussian_sigma)
    #     ipsf = utils.fftconvolution(
    #         sample=modelpsfgen.ipsf,
    #         kernel=kernel,
    #         boundary='fill',
    #         nan_treatment='fill',
    #         fill_value=0,
    #         normalize_kernel=np.sum
    #     ).astype(np.float32)
    #     ipsf /= np.max(ipsf)
    #
    #     iotf = utils.fft(ipsf)
    #     iotf = utils.normalize_otf(iotf)
    #
    # else:
    #     iotf = modelpsfgen.iotf
    
    if fov_is_small:  # only going to center crop and predict on that single FOV (fourier_embeddings)

        if not skip_prep_sample:
            sample = prep_sample(
                sample,
                model_fov=modelpsfgen.psf_fov,              # this is what we will crop to
                sample_voxel_size=samplepsfgen.voxel_size,
                remove_background=remove_background,
                normalize=normalize,
                read_noise_bias=read_noise_bias,
                min_psnr=min_psnr,
                plot=plot if plot else None,
                na_mask=samplepsfgen.na_mask,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size,
            )
        
        embs = fourier_embeddings(
            sample,
            iotf=modelpsfgen.iotf,
            na_mask=modelpsfgen.na_mask,
            plot=plot if plot else None,
            no_phase=no_phase,
            remove_interference=True,
            embedding_option=modelpsfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            model_psf_shape=modelpsfgen.psf_shape,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma,
            use_reconstructed_otf=use_reconstructed_otf
        ).astype(np.float32)
        
        if save_processed_tif_file:
            imwrite(
                file,
                data=embs.astype(np.float32),
                compression='deflate',
                dtype=np.float32,
                imagej=True,
                metadata={'axes': 'TZYXS'}
            )
            return file
        else:
            return embs

    else:  # at least one tile fov dimension is larger than model fov
        if interpolate_embeddings:
            if not skip_prep_sample:
                voxel_size_scalar = np.array(modelpsfgen.voxel_size) ** 2 / np.array(samplepsfgen.voxel_size)
                adjusted_pov = tuple(
                    voxel_size_scalar * np.array(sample.shape) * np.array(samplepsfgen.fov_scaler)
                )
                
                sample = prep_sample(
                    sample,
                    model_fov=adjusted_pov,
                    sample_voxel_size=samplepsfgen.voxel_size,
                    remove_background=remove_background,
                    normalize=normalize,
                    read_noise_bias=read_noise_bias,
                    min_psnr=min_psnr,
                    plot=plot if plot else None,
                    na_mask=samplepsfgen.na_mask,
                    denoiser=denoiser,
                    denoiser_window_size=denoiser_window_size,
                )
            
            embs = fourier_embeddings(
                sample,
                iotf=modelpsfgen.iotf,
                na_mask=modelpsfgen.na_mask,
                plot=plot if plot else None,
                no_phase=no_phase,
                remove_interference=True,
                embedding_option=modelpsfgen.embedding_option,
                freq_strength_threshold=freq_strength_threshold,
                digital_rotations=digital_rotations,
                model_psf_shape=modelpsfgen.psf_shape,
                interpolate_embeddings=interpolate_embeddings,
                estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
            ).astype(np.float32)
            
            if save_processed_tif_file:
                imwrite(
                    file,
                    data=sample.astype(np.float32),
                    compression='deflate',
                    dtype=np.float32,
                    imagej=True,
                    metadata={'axes': 'TZYXS'}
                )
                return file
            else:
                return embs

        
        else:  # average FFTs
            model_window_size = (
                round_to_even(modelpsfgen.psf_fov[0] / samplepsfgen.voxel_size[0]),
                round_to_even(modelpsfgen.psf_fov[1] / samplepsfgen.voxel_size[1]),
                round_to_even(modelpsfgen.psf_fov[2] / samplepsfgen.voxel_size[2]),
            )  # number of sample voxels that make up a model psf.
            
            model_window_size = np.minimum(model_window_size, sample.shape)
            
            # how many non-overlapping rois can we split this tile fov into (round down)
            rois = sliding_window_view(
                sample,
                window_shape=model_window_size
            )  # stride = 1.
            
            # decimate from all available windows to the stride we want (either rolling_strides or model_window_size)
            if rolling_strides is not None:
                strides = rolling_strides
            else:
                strides = model_window_size
            
            rois = rois[
                   ::strides[0],
                   ::strides[1],
                   ::strides[2]
                   ]
            
            throwaway = np.array(sample.shape) - ((np.array(rois.shape[:3]) - 1) * strides + rois.shape[-3:])
            if any(throwaway > (np.array(sample.shape) / 4)):
                raise Exception(f'You are throwing away {throwaway} voxels out of {sample.shape}, with stride length'
                                f'{strides}. Change rolling_strides.')
            
            ztiles, nrows, ncols = rois.shape[:3]
            rois = np.reshape(rois, (-1, *model_window_size))
            
            if not skip_prep_sample:
                prep = partial(
                    prep_sample,
                    sample_voxel_size=samplepsfgen.voxel_size,
                    remove_background=remove_background,
                    normalize=normalize,
                    read_noise_bias=read_noise_bias,
                    na_mask=samplepsfgen.na_mask,
                    min_psnr=min_psnr,
                    denoiser=denoiser,
                    denoiser_window_size=denoiser_window_size,
                )
                
                rois = utils.multiprocess(
                    func=prep,
                    jobs=rois,
                    desc=f'Preprocessing, {rois.shape[0]} rois per tile, '
                         f'roi size, {rois.shape[-3:]}, '
                         f'stride length {strides}, '
                         f'throwing away {throwaway} voxels'
                )
            
            embs = rolling_fourier_embeddings(  # aka "large_fov"
                rois,
                iotf=modelpsfgen.iotf,
                na_mask=modelpsfgen.na_mask,
                plot=plot,
                no_phase=no_phase,
                remove_interference=True,
                embedding_option=modelpsfgen.embedding_option,
                freq_strength_threshold=freq_strength_threshold,
                digital_rotations=digital_rotations,
                model_psf_shape=modelpsfgen.psf_shape,
                nrows=nrows,
                ncols=ncols,
                ztiles=ztiles,
                cpu_workers=cpu_workers,
                estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
            ).astype(np.float32)
            
            if save_processed_tif_file:
                imwrite(
                    file,
                    data=sample.astype(np.float32),
                    compression='deflate',
                    dtype=np.float32,
                    imagej=True,
                    metadata={'axes': 'TZYXS'}
                )
                return file
            else:
                return embs



@profile
def bootstrap_predict(
        model: tf.keras.Model,
        inputs: np.array,
        psfgen: SyntheticPSF,
        batch_size: int = 512,
        n_samples: int = 1,
        no_phase: bool = False,
        padsize: Any = None,
        alpha_val: str = 'abs',
        phi_val: str = 'angle',
        pois: Any = None,
        remove_interference: bool = True,
        ignore_modes: list = (0, 1, 2, 4),
        threshold: float = 0.,
        freq_strength_threshold: float = .01,
        verbose: int = 1,
        plot: Any = None,
        desc: str = 'MiniBatch-probabilistic-predictions',
        cpu_workers: int = 1
):
    """
    Average predictions and compute stdev

    Args:
        model: pre-trained keras model
        inputs: encoded tokens to be processed
        psfgen: Synthetic PSF object
        n_samples: number of predictions of average
        ignore_modes: list of modes to ignore
        batch_size: number of samples per batch
        no_phase: ignore/drop the phase component of the FFT
        padsize: pad the input to the desired size for the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        remove_interference: a toggle to normalize out the interference pattern from the OTF
        pois: masked array of the peaks of interest to compute the interference pattern
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        desc: test to display for the progressbar
        verbose: a toggle for progress bar
        plot: optional toggle to visualize embeddings

    Returns:
        average prediction, stdev
    """
    threshold = utils.waves2microns(threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))
    logger.info(f"Ignoring modes: {ignore_modes}")

    # check z-axis to compute embeddings for fourier models
    if len(inputs.shape) == 3:
        emb = model.input_shape[1] == inputs.shape[0]
    else:
        emb = model.input_shape[1] == inputs.shape[1]

    if not emb:
        model_inputs = fourier_embeddings(
            inputs,
            iotf=psfgen.iotf,
            na_mask=psfgen.na_mask,
            plot=plot,
            padsize=padsize,
            no_phase=no_phase,
            alpha_val=alpha_val,
            phi_val=phi_val,
            pois=pois,
            remove_interference=remove_interference,
            embedding_option=psfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
        )  # (nx361, 6, 64, 64, 1) n=# of samples (e.g. # of image vols)
    else:
        # pass raw PSFs to the model
        model_inputs = inputs

    logger.info(f"Checking for invalid inputs")
    model_inputs = np.nan_to_num(model_inputs, nan=0, posinf=0, neginf=0)
    model_inputs = model_inputs[..., np.newaxis] if model_inputs.shape[-1] != 1 else model_inputs
    features = np.array([np.count_nonzero(s) for s in inputs])

    logger.info(f"[BS={batch_size}, n={n_samples}] {desc}")
    # batch_size is over number of samples (e.g. # of image vols)
    gen = tf.data.Dataset.from_tensor_slices(model_inputs).batch(batch_size).repeat(n_samples)  # (None, 6, 64, 64, 1)
    preds = model.predict(gen, verbose=verbose)

    if preds.shape[1] > 1:
        preds[:, ignore_modes] = 0.

    preds[np.abs(preds) <= threshold] = 0.
    preds = np.stack(np.split(preds, n_samples), axis=-1)

    mu = np.mean(preds, axis=-1)
    mu = mu.flatten() if mu.shape[0] == 1 else mu

    sigma = np.std(preds, axis=-1)
    sigma = sigma.flatten() if sigma.shape[0] == 1 else sigma

    mu[np.where(features == 0)[0]] = np.zeros_like(mu[0])
    sigma[np.where(features == 0)[0]] = np.zeros_like(sigma[0])

    return mu, sigma


@profile
def eval_rotation(
        init_preds: np.ndarray,
        rotations: Union[int, np.ndarray],
        psfgen: SyntheticPSF,
        save_path: Path,
        plot: Any = None,
        threshold: float = 0.,
        no_phase: bool = False,
        confidence_threshold: float = .02,
        minimum_fraction_of_kept_points: float = 0.45,
):
    """  # order matters future thayer (predict_dataset)
        We can think of the mode and its twin as the X and Y basis, and the aberration being a
        vector on this coordinate system. A problem occurs when the ground truth vector lies near
        where one of the vectors flips sign (and the model was trained to only respond with positive
        values for that vector).  Either a discontinuity or a degeneracy occurs in this area leading to
        unreliable predictions.  The solution is to digitally rotate the input embeddings over a range
        of angles.  Converting from cartesian X Y to polar coordinates (rho, phi), the model predictions
        should map phi as a linear waveform (if all were perfect), a triangle waveform (if degenergancies
        exist because the model was trained to only return positive values), or a sawtooth waveform (if
        a discontinuity exists because the model was trained to only return positive values for the first
        mode in the twin pair).  These waveforms (with known period:given by the m value of the mode,
        known amplitude) can be curve-fit to determine the actual phi. Given rho and phi, we can
        return to cartesian coordinates, which give the amplitudes for mode & twin.

        Args:
            init_preds: predictions for each rotation angle
            rotations: list of rotations applied to the embeddings
            psfgen: Synthetic PSF object
            threshold: set predictions below threshold to zero (wavelength)
            plot: optional toggle to visualize embeddings
    """

    def cart2pol(x, y):
        """Convert cartesian (x, y) to polar (rho, phi_in_radians)
        """
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)  # Note role reversal: the "y-coordinate" is 1st parameter, the "x-coordinate" is 2nd.
        return (rho, phi)

    def pol2cart(rho, phi):
        """Convert polar (rho, phi) to cartesian (x, y)
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def linear_fit_fixed_slope(x, y, m):
        return np.mean(y - m * x)

    threshold = utils.waves2microns(threshold, wavelength=psfgen.lam_detection)

    if isinstance(rotations, int):
        rotations = np.linspace(0, 360, rotations)

    preds = np.zeros(psfgen.n_modes)
    stdevs = np.zeros(psfgen.n_modes)
    wavefront = Wavefront(preds)
    mses = np.zeros(len(wavefront.twins.items()))
    results = []

    for row, (mode, twin) in enumerate(wavefront.twins.items()):
        df = pd.DataFrame(rotations, columns=['angle'])
        df['mode'] = mode.index_ansi
        df['twin'] = mode.index_ansi if twin is None else twin.index_ansi

        df['init_pred_mode'] = init_preds[:, mode.index_ansi]
        df['init_pred_twin'] = init_preds[:, mode.index_ansi] if twin is None else init_preds[:, twin.index_ansi]

        if twin is not None:
            # the Zernike modes have m periods per 2pi. xdata is now the Twin angle
            rhos, ydata = cart2pol(init_preds[:, mode.index_ansi], init_preds[:, twin.index_ansi])

            xdata = rotations * np.abs(mode.m)
            df['twin_angle'] = xdata

            ydata = np.degrees(ydata) #% 180 # so now agnostic to direction

            rho = rhos[np.argmin(np.abs(ydata))]
            std_rho = np.std(rhos).round(3)
            df['rhos'] = rhos

            # if spatial model: exclude points near discontinuities (-90, +90, 450,..) based upon fit
            data_mask = np.ones(xdata.shape[0], dtype=bool)

            if no_phase:
                data_mask[
                    np.abs(init_preds[:, mode.index_ansi] / rho) < np.cos(np.radians(70)) * (rho > threshold)
                ] = 0.

            # exclude if rho is unusually small
            # (which can lead to small, but dominant primary mode near discontinuity)
            data_mask[rhos < np.mean(rhos) - np.std(rhos)] = 0.
            df['valid_points'] = np.ones(xdata.shape[0], dtype=bool)  # data_mask

            xdata_masked = xdata[data_mask]
            ydata_masked = ydata[data_mask]
            offset = ydata_masked[0]
            ydata_masked = np.unwrap(ydata_masked, period=180)
            # put start between -90 and +90
            ydata_masked = ((ydata_masked - offset - xdata_masked) + 90) % 180 - 90 + offset + xdata_masked

            m = 1
            b = linear_fit_fixed_slope(xdata_masked, ydata_masked, m)  # refit without bad data points
            fit = m * xdata + b
            fit_p180 = m * xdata + (b + 180)

            number_of_wraps = (fit - ydata) // 180
            below = ydata + (number_of_wraps * 180)
            above = ydata + ((number_of_wraps+1) * 180)
            ydata = above
            below_is_better = np.abs(fit-below) < np.abs(fit-above)
            ydata[below_is_better] = below[below_is_better]

            junk, twin_angles = cart2pol(init_preds[:, mode.index_ansi], init_preds[:, twin.index_ansi])
            twin_angles = np.degrees(twin_angles)
            leave_sign_votes_for = np.abs(angle_diff(twin_angles, fit)) < np.abs(angle_diff(twin_angles, fit_p180))
            leave_sign_votes_for = np.sum(leave_sign_votes_for[data_mask].astype(int))
            leave_sign_votes_against = len(twin_angles[data_mask]) - leave_sign_votes_for

            if leave_sign_votes_for > leave_sign_votes_against:
                fitted_twin_angle_b = b  # evaluate the curve fit when there is no digital rotation.
                df['swapped_sign'] = False
            else:
                fitted_twin_angle_b = (b + 180) % 360  # evaluate the curve fit when there is no digital rotation.
                df['swapped_sign'] = True

            df['pred_twin_angle'] = ydata
            df['fitted_twin_angle'] = m * df['twin_angle'] + b
            df['fitted_twin_angle_b'] = fitted_twin_angle_b

            squared_error = np.square(fit - ydata)
            mse = np.mean(squared_error)
            if mse > 700:
                # reject if it doesn't show rotation that matches within +/- 26deg,
                # equivalent to +/- 0.005 um on a 0.01 um mode.
                rho = 0
                confident = all(rhos < confidence_threshold) # either confident zero or unconfident
            else:
                rho = np.mean(rhos[squared_error < 700])
                confident = rho / std_rho > 1 # is SNR above 1? # either confident-A or unconfident

            df['mse'] = mse


            """
                rho is already set to zero if `fraction_of_kept_points` and/or `mse` are bad
                                    prediction  stdev
                confident-A         p           s (small)
                confident-Z         0           s (small)
                unconfident         0           0
            """

            if np.allclose(rhos, rhos[0], atol=0.0001):  # blank image (unconfident)
                preds[mode.index_ansi], preds[twin.index_ansi] = 0., 0.
                stdevs[mode.index_ansi], stdevs[twin.index_ansi] = 0., 0.
                confident = 0.
            elif confident and rho > 0:  # confident-A
                preds[mode.index_ansi], preds[twin.index_ansi] = pol2cart(rho, np.radians(fitted_twin_angle_b))
                stdevs[mode.index_ansi], stdevs[twin.index_ansi] = std_rho, std_rho
            elif confident and rho == 0:  # confident-Z
                preds[mode.index_ansi], preds[twin.index_ansi] = 0., 0.
                stdevs[mode.index_ansi], stdevs[twin.index_ansi] = std_rho, std_rho
            else:  # unconfident
                preds[mode.index_ansi], preds[twin.index_ansi] = 0., 0.
                stdevs[mode.index_ansi], stdevs[twin.index_ansi] = 0., 0.

        else:  # mode has m=0 (spherical,...), or twin isn't within the 55 modes.

            rhos = init_preds[:, mode.index_ansi]
            std_rho = np.std(rhos)

            if np.allclose(rhos, rhos[0], atol=0.0001):  # blank image (unconfident)
                preds[mode.index_ansi] = 0.
                stdevs[mode.index_ansi] = 0.
                rhos = np.zeros_like(rhos)
                confident = 0.
            else:
                rho = np.median(rhos)
                rho *= np.abs(rho) > threshold  # keep it if it's above threshold, or else set to zero.

                if all(np.abs(rhos) < confidence_threshold):
                    confident = 1
                    rho = 0  # confident-Z
                else:
                    confident = np.abs(rho) / std_rho > 4   # is SNR above 4?

                if confident and np.abs(rho) > 0:  # confident-A
                    preds[mode.index_ansi] = rho
                    stdevs[mode.index_ansi] = std_rho
                elif confident and rho == 0:  # confident-Z
                    preds[mode.index_ansi] = 0.
                    stdevs[mode.index_ansi] = std_rho
                else:  # unconfident
                    preds[mode.index_ansi] = 0.
                    stdevs[mode.index_ansi] = 0.

            df['rhos'] = rhos
            df['valid_points'] = 1
            df['twin_angle'] = np.nan
            df['pred_twin_angle'] = np.nan
            df['fitted_twin_angle'] = np.nan
            df['mse'] = np.nan


        df['confident'] = confident
        df['aggr_rho'] = rho
        df['aggr_mode_amp'] = preds[mode.index_ansi]
        df['aggr_twin_amp'] = np.nan if twin is None else preds[twin.index_ansi]
        df['aggr_std_dev'] = std_rho
        df['aggr_twin_std_dev'] = np.nan if twin is None else std_rho

        results.append(df)

    results = pd.concat(results, ignore_index=True)
    try:
        results.to_csv(Path(f'{save_path}_rotations.csv'))
    except PermissionError:
        logger.error(f'Permission denied: {save_path.resolve()}_rotations.csv')

    if plot is not None:
        vis.plot_rotations(Path(f'{plot}_rotations.csv'))
        
        p = Wavefront(preds, order='ansi', lam_detection=psfgen.lam_detection)
        std = Wavefront(stdevs, order='ansi', lam_detection=psfgen.lam_detection)
        
        predicted_psf = psfgen.single_psf(phi=p, normed=True, lls_defocus_offset=0.)
        predicted_embeddings = fourier_embeddings(
            predicted_psf,
            iotf=psfgen.iotf,
            na_mask=psfgen.na_mask,
            remove_interference=False
        )
        
        vis.diagnosis(
            pred=p,
            pred_std=std,
            save_path=Path(f"{plot}_diagnosis"),
            lls_defocus=0.,
            predicted_psf=predicted_psf,
            predicted_embeddings=predicted_embeddings
        )
        
    return preds, stdevs


def angle_diff(a, b):
    return np.degrees(np.arctan2(np.sin(np.radians(a - b)), np.cos(np.radians(a - b))))


@profile
def predict_rotation(
        model: tf.keras.Model,
        inputs: np.ndarray,
        psfgen: SyntheticPSF,
        save_path: Path,
        batch_size: int = 128,
        no_phase: bool = False,
        padsize: Any = None,
        alpha_val: str = 'abs',
        phi_val: str = 'angle',
        pois: Any = None,
        ignore_modes: list = (0, 1, 2, 4),
        threshold: float = 0.,
        freq_strength_threshold: float = .01,
        plot: Any = None,
        plot_rotations: Any = None,
        remove_interference: bool = True,
        desc: str = 'Predict-rotations',
        confidence_threshold: float = .02,
        digital_rotations: Optional[int] = 361,
        cpu_workers: int = -1,
):
    """
    Predict the fraction of the amplitude to be assigned each pair of modes (ie. mode & twin).

    Args:
        model: pre-trained keras model. Model must be trained with XY embeddings only so that we can rotate them.
        inputs: encoded tokens to be processed. (e.g. input images)
        psfgen: Synthetic PSF object
        batch_size: number of samples per batch
        no_phase: ignore/drop the phase component of the FFT
        padsize: pad the input to the desired size for the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        remove_interference: a toggle to normalize out the interference pattern from the OTF
        pois: masked array of the peaks of interest to compute the interference pattern
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        desc: test to display for the progressbar
        plot: optional toggle to visualize embeddings
    """
    # check z-axis to compute embeddings for fourier models
    if len(inputs.shape) == 3:
        emb = model.input_shape[1] == inputs.shape[0]
    else:
        emb = model.input_shape[1] == inputs.shape[1]

    if not emb:
        inputs = fourier_embeddings(
            inputs,
            iotf=psfgen.iotf,
            na_mask=psfgen.na_mask,
            # plot=plot,
            padsize=padsize,
            no_phase=no_phase,
            alpha_val=alpha_val,
            phi_val=phi_val,
            pois=pois,
            remove_interference=remove_interference,
            embedding_option=psfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations
        )  # inputs.shape = (361 rotations, 6 embeddings, 64, 64, 1)

    init_preds, stdev = bootstrap_predict(
        model,
        inputs,
        psfgen=psfgen,
        batch_size=batch_size,
        n_samples=1,
        no_phase=no_phase,
        threshold=0.,
        ignore_modes=ignore_modes,
        plot=plot,
        desc=desc,
        cpu_workers=cpu_workers
    )

    eval_mode_rotations = partial(
        eval_rotation,
        rotations=digital_rotations,
        psfgen=psfgen,
        threshold=threshold,
        plot=plot_rotations,
        no_phase=no_phase,
        confidence_threshold=confidence_threshold,
        save_path=save_path,
    )

    init_preds = np.stack(np.split(init_preds, digital_rotations), axis=1)

    if init_preds.shape[-1] > 1:

        jobs = utils.multiprocess(
            jobs=init_preds,
            func=eval_mode_rotations,
            cores=cpu_workers,
            desc="Evaluate predictions"
        )
        jobs = np.array([list(zip(*j)) for j in jobs])
        preds, stdev = jobs[..., 0], jobs[..., -1]

        if init_preds.shape[-1] == psfgen.n_modes:
            return preds, stdev
        else:
            lls_defocus = np.mean(init_preds[:, -1])
            return preds, stdev, lls_defocus

    else:
        preds = np.zeros(psfgen.n_modes)
        stdev = np.zeros(psfgen.n_modes)
        lls_defocus = np.mean(init_preds)
        return preds, stdev, lls_defocus


@profile
def predict_sign(
        init_preds: np.ndarray,
        followup_preds: np.ndarray,
        sign_threshold: float = .99,
        plot: Any = None,
):
    def pct_change(cur, prev):
        cur = np.abs(cur)
        prev = np.abs(prev)

        if np.array_equal(cur, prev):
            return np.zeros_like(prev)
        pct = ((cur - prev) / (prev + 1e-6)) * 100.0
        pct[pct > 100] = 100
        pct[pct < -100] = -100
        return pct

    preds = init_preds.copy()
    pchange = pct_change(followup_preds, init_preds)

    # flip signs and make any necessary adjustments to the amplitudes based on the followup predictions
    # adj = pchange.copy()
    # adj[np.where(pchange > 0)] = -200
    # adj[np.where(pchange < 0)] += 100
    # preds += preds * (adj/100)

    # threshold-based sign prediction
    # threshold = sign_threshold * init_preds
    # flips = np.stack(np.where(followup_preds > threshold), axis=0)

    # if len(np.squeeze(preds).shape) == 1:
    #     preds[flips[0]] *= -1
    # else:
    #     preds[flips[0], flips[1]] *= -1

    # flip sign only if amplitudes increased on the followup predictions
    preds[pchange > 0.] *= -1

    if plot is not None:
        if len(np.squeeze(preds).shape) == 1:
            init_preds_wave = np.squeeze(init_preds)
            init_preds_wave_error = np.zeros_like(init_preds_wave)

            followup_preds_wave = np.squeeze(followup_preds)
            followup_preds_wave_error = np.zeros_like(followup_preds_wave)

            preds_wave = np.squeeze(preds)
            preds_error = np.zeros_like(preds_wave)

            percent_changes = np.squeeze(pchange)
            percent_changes_error = np.zeros_like(percent_changes)
        else:
            init_preds_wave = np.mean(init_preds, axis=0)
            init_preds_wave_error = np.std(init_preds, axis=0)

            followup_preds_wave = np.mean(followup_preds, axis=0)
            followup_preds_wave_error = np.std(followup_preds, axis=0)

            preds_wave = np.mean(preds, axis=0)
            preds_error = np.std(preds, axis=0)

            percent_changes = np.mean(pchange, axis=0)
            percent_changes_error = np.std(pchange, axis=0)

        plt.style.use("default")
        vis.sign_correction(
            init_preds_wave,
            init_preds_wave_error,
            followup_preds_wave,
            followup_preds_wave_error,
            preds_wave,
            preds_error,
            percent_changes,
            percent_changes_error,
            savepath=f'{plot}_sign_correction'
        )

        plt.style.use("dark_background")
        vis.sign_correction(
            init_preds_wave,
            init_preds_wave_error,
            followup_preds_wave,
            followup_preds_wave_error,
            preds_wave,
            preds_error,
            percent_changes,
            percent_changes_error,
            savepath=f'{plot}_sign_correction_db'
        )

    return preds, pchange


@profile
def dual_stage_prediction(
        model: tf.keras.Model,
        inputs: np.array,
        gen: SyntheticPSF,
        modelgen: SyntheticPSF,
        plot: Any = None,
        verbose: bool = False,
        threshold: float = 0.,
        freq_strength_threshold: float = .01,
        sign_threshold: float = .9,
        n_samples: int = 1,
        batch_size: int = 1,
        ignore_modes: list = (0, 1, 2, 4),
        prev_pred: Any = None,
        estimate_sign_with_decon: bool = False,
        decon_iters: int = 5
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    prev_pred = None if (prev_pred is None or str(prev_pred) == 'None') else prev_pred

    res = predict_rotation(
        model,
        inputs,
        psfgen=modelgen,
        no_phase=True,
        verbose=verbose,
        batch_size=batch_size,
        threshold=threshold,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        plot=plot,
        plot_rotations=plot,
    )

    try:
        init_preds, stdev = res
    except ValueError:
        init_preds, stdev, lls_defocus = res

    if estimate_sign_with_decon:
        logger.info(f"Estimating signs w/ Decon")
        abrs = range(init_preds.shape[0]) if len(init_preds.shape) > 1 else range(1)
        make_psf = partial(gen.single_psf, normed=True)
        psfs = np.stack(gen.batch(
            make_psf,
            [Wavefront(init_preds[i], lam_detection=gen.lam_detection) for i in abrs]
        ), axis=0)

        with sp.fft.set_workers(-1):
            followup_inputs = richardson_lucy(np.squeeze(inputs), np.squeeze(psfs), num_iter=decon_iters)

        if len(followup_inputs.shape) == 3:
            followup_inputs = followup_inputs[np.newaxis, ..., np.newaxis]
        else:
            followup_inputs = followup_inputs[..., np.newaxis]

        res = predict_rotation(
            model,
            followup_inputs,
            psfgen=modelgen,
            no_phase=True,
            verbose=False,
            batch_size=batch_size,
            threshold=threshold,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
        )

        try:
            followup_preds, stdev = res
        except ValueError:
            followup_preds, stdev, lls_defocus = res

        preds, pchanges = predict_sign(
            init_preds=init_preds,
            followup_preds=followup_preds,
            sign_threshold=sign_threshold,
            plot=plot
        )

    elif prev_pred is not None:
        logger.info(f"Evaluating signs")
        followup_preds = init_preds.copy()
        init_preds = pd.read_csv(prev_pred, header=0)['amplitude'].values

        preds, pchanges = predict_sign(
            init_preds=init_preds,
            followup_preds=followup_preds,
            sign_threshold=sign_threshold,
            plot=plot
        )

    else:
        preds = init_preds
        pchanges = np.zeros_like(preds)

    return preds, stdev, pchanges


@profile
def predict_dataset(
        model: Union[tf.keras.Model, Path],
        inputs: tf.data.Dataset,
        psfgen: SyntheticPSF,
        save_path: list,
        batch_size: int = 128,
        ignore_modes: list = (0, 1, 2, 4),
        threshold: float = 0.,
        confidence_threshold: float = .02,
        verbose: int = 1,
        desc: str = 'MiniBatch-probabilistic-predictions',
        digital_rotations: Optional[int] = None,
        plot_rotations: Any = None,
        pool: Optional[mp.Pool] = None,
        cpu_workers: int = -1,
):
    """
    Average predictions and compute stdev

    Args:
        model: pre-trained keras model
        inputs: encoded tokens to be processed
        psfgen: Synthetic PSF object
        ignore_modes: list of modes to ignore
        batch_size: number of samples per batch
        threshold: set predictions below threshold to zero (wavelength)
        desc: test to display for the progressbar
        verbose: a toggle for progress bar
        digital_rotations: an array of digital rotations to apply to evaluate model's confidence
        plot_rotations: optional toggle to plot digital rotations

    Returns:
        average prediction, stdev
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    if isinstance(model, Path):
        model = load(model)

    no_phase = True if model.input_shape[1] == 3 else False
    threshold = utils.waves2microns(threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))
    logger.info(f"Ignoring modes: {ignore_modes}")
    logger.info(desc)

    if digital_rotations is not None:
        inputs = inputs.map(lambda x: tf.reshape(x, shape=(-1, *model.input_shape[1:]))).unbatch()
        inputs = inputs.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        inputs = inputs.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(inputs, verbose=verbose)
    preds[:, ignore_modes] = 0.
    preds[np.abs(preds) <= threshold] = 0.

    if digital_rotations is not None:
        with pool if pool is not None else mp.Pool(processes=cpu_workers if cpu_workers > 0 else mp.cpu_count()) as p:
            preds = np.array(np.split(preds, len(save_path)))
            jobs = list(tqdm(
                p.starmap(
                    eval_rotation,  # order matters future thayer
                    zip(
                        preds,  # init_preds
                        repeat(digital_rotations),                      # rotations
                        repeat(psfgen),                                 # psfgen
                        save_path,                                      # save_path
                        save_path if plot_rotations else repeat(None),  # plot
                        repeat(threshold),                              # threshold
                        repeat(no_phase),                               # no_phase
                        repeat(confidence_threshold),                   # confidence_threshold
                    ),
                ),
                total=len(save_path),
                desc="Evaluate predictions",
                unit=' evals',
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                file=sys.stdout,
            ))

        jobs = np.array([list(zip(*j)) for j in jobs])
        preds, std = jobs[..., 0], jobs[..., -1]
        return preds, std

    else:
        return preds, np.zeros_like(preds)


@profile
def predict_files(
    paths: np.ndarray,
    outdir: Path,
    model: tf.keras.Model,
    modelpsfgen: SyntheticPSF,
    samplepsfgen: Optional[SyntheticPSF] = None,
    dm_calibration: Optional[Union[Path, str]] = None,
    dm_state: Optional[Union[Path, str, np.array]] = None,
    wavelength: float = .510,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .02,
    batch_size: int = 1,
    digital_rotations: Optional[int] = 361,
    rolling_strides: Optional[tuple] = None,
    fov_is_small: bool = True,
    plot: bool = False,
    plot_rotations: bool = False,
    skip_prep_sample: bool = False,
    preprocessed: bool = False,
    remove_background: bool = True,
    cpu_workers: int = -1,
    template: Optional[pd.DataFrame] = None,
    pool: Optional[mp.Pool] = None,
    min_psnr: int = 5,
    estimated_object_gaussian_sigma: float = 0,
    save_processed_tif_file: bool = False
):
    if not preprocessed:
        generate_fourier_embeddings = partial(
            preprocess,
            modelpsfgen=modelpsfgen,
            samplepsfgen=samplepsfgen,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            plot=plot,
            no_phase=True if model.input_shape[1] == 3 else False,
            remove_background=remove_background,
            normalize=True,
            fov_is_small=fov_is_small,
            rolling_strides=rolling_strides,
            skip_prep_sample=skip_prep_sample,
            min_psnr=min_psnr,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma,
            cpu_workers=1,  # already parallelized over files
            save_processed_tif_file=save_processed_tif_file
        )
        
        paths = multiprocess(
            func=generate_fourier_embeddings,
            jobs=paths,
            cores=cpu_workers,
            desc='Generate fourier embeddings',
            unit=' .tif file'
        )
    
    inputs = tf.data.Dataset.from_tensor_slices(np.vectorize(str)(paths))
    
    inputs = inputs.map(
        lambda x: tf.py_function(
            load_sample,
            inp=[x],
            Tout=tf.float32,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    
    # inputs = tf.data.Dataset.from_generator(
    #     DatasetGenerator(paths=paths, load_preprocess_func=generate_fourier_embeddings),
    #     output_types=(tf.float32),
    #     output_shapes=(digital_rotations, *model.input_shape[1:])
    # )

    num_predictions = paths.shape[0] * digital_rotations if digital_rotations else paths.shape[0]

    desc = ' '.join([
        f"[{paths.shape[0]} ROIs] x [{digital_rotations} Rotations] =",
        f"{num_predictions} predictions,",
        f"requires {int(np.ceil(num_predictions / batch_size))} inference batches ({batch_size=}).",
        # f"emb={6 * 64 * 64 * batch_size * digital_rotations if digital_rotations else 1 * 32 / 8 / 1e6:.2f} MB/batch.",
    ])

    preds, std = predict_dataset(
        model,
        inputs=inputs,
        psfgen=modelpsfgen,
        batch_size=batch_size,
        ignore_modes=ignore_modes,
        threshold=prediction_threshold,
        save_path=[f.with_suffix('') for f in paths],
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        confidence_threshold=confidence_threshold,
        desc=desc,
        cpu_workers=cpu_workers,
        pool=None,  # let the "eval_tiles" pool spawn just before predict is called
    )

    if template is not None:
        # initialize predictions dataframe with NaNs, the ignored tiles (dropped from poor SNR) will remain as NaNs.
        # NaNs in the _predictions.csv will appear as blank.
        # ignored tiles will be written to _settings.json and read in from aggregate from there.
        tile_names = [f.stem for f in paths]
        predictions = pd.DataFrame(
            np.full((preds.shape[1], template.shape[1]), fill_value=np.NaN),
            columns=template.columns
        )
        predictions[tile_names] = preds.T   # overwrite with predictions

        stdevs = pd.DataFrame(np.zeros((std.shape[1], template.shape[1])), columns=template.columns)
        stdevs[tile_names] = std.T

    else:
        tile_names = [f.stem for f in paths]
        predictions = pd.DataFrame(preds.T, columns=tile_names)
        stdevs = pd.DataFrame(std.T, columns=tile_names)

    if dm_calibration is not None:
        zernikies_to_actuators = partial(utils.zernikies_to_actuators, dm_calibration=dm_calibration, dm_state=dm_state)
        actuators = predictions.apply(lambda x: zernikies_to_actuators(x), axis=0)

        actuators.index.name = 'actuators'
        actuators.to_csv(f"{outdir}_predictions_corrected_actuators.csv")

    predictions['mean'] = predictions[tile_names].mean(axis=1)
    predictions['median'] = predictions[tile_names].median(axis=1)
    predictions['min'] = predictions[tile_names].min(axis=1)
    predictions['max'] = predictions[tile_names].max(axis=1)
    predictions['std'] = predictions[tile_names].std(axis=1)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{outdir}_predictions.csv")

    stdevs['mean'] = stdevs[tile_names].mean(axis=1)
    stdevs['median'] = stdevs[tile_names].median(axis=1)
    stdevs['min'] = stdevs[tile_names].min(axis=1)
    stdevs['max'] = stdevs[tile_names].max(axis=1)
    stdevs['std'] = stdevs[tile_names].std(axis=1)
    stdevs.index.name = 'ansi'
    stdevs.to_csv(f"{outdir}_stdevs.csv")

    return predictions, stdevs
