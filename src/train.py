import matplotlib
matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import warnings
warnings.filterwarnings("ignore")

import contextlib
import logging
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.optimizers.schedules import CosineDecay

from callbacks import Defibrillator
from callbacks import TensorBoardCallback
from backend import load

import utils
import data_utils
import aovift
import vit
import convnext
import otfnet
import prototype
import cli

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)
plt.set_loglevel('error')


def plot_conv_patches(name: str, img: np.ndarray, outdir: Path, patch_ize: int):
    input_img = np.expand_dims(img, axis=0)
    vmin = np.min(input_img)
    vmax = np.max(input_img)
    cmap = "Spectral"
    
    for k, label in enumerate(['xy', 'xz', 'yz']):
        original = np.squeeze(img[k])

        plt.figure(figsize=(4, 4))
        plt.imshow(original, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title('Original')
        plt.savefig(f'{outdir}/{name}_{label}_original.png', dpi=300, bbox_inches='tight', pad_inches=.25)

        patches = aovift.Patchify(patch_size=patch_ize)(input_img)
        patches = patches[0, k]

        n = int(np.sqrt(patches.shape[0]))
        plt.figure(figsize=(4, 4))
        plt.title('Patches')
        for i, patch in enumerate(patches):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_ize, patch_ize)).numpy()
            ax.imshow(patch_img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
        
        plt.axis('off')
        plt.savefig(f'{outdir}/{name}_{label}_patches_p{patch_ize}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_multiscale_patches(name: str, img: np.ndarray, outdir: Path, patches: list):
    input_img = np.expand_dims(img, axis=0)
    vmin = np.min(input_img)
    vmax = np.max(input_img)
    cmap = "Spectral"
    
    for k, label in enumerate(['xy', 'xz', 'yz']):
        original = np.squeeze(img[k])

        plt.figure(figsize=(4, 4))
        plt.imshow(original, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title('Original')
        plt.savefig(f'{outdir}/{name}_{label}_original.png', dpi=300, bbox_inches='tight', pad_inches=.25)

        for p in patches:
            patches = aovift.Patchify(patch_size=p)(input_img)
            merged = aovift.Merge(patch_size=p)(patches)

            patches = patches[0, k]
            merged = np.squeeze(merged[0, k])

            plt.figure(figsize=(4, 4))
            plt.imshow(merged, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.title('Merged')
            plt.savefig(f'{outdir}/{label}_merged.png', dpi=300, bbox_inches='tight', pad_inches=.25)

            n = int(np.sqrt(patches.shape[0]))
            plt.figure(figsize=(4, 4))
            plt.title('Patches')
            for i, patch in enumerate(patches):
                ax = plt.subplot(n, n, i + 1)
                patch_img = tf.reshape(patch, (p, p)).numpy()
                ax.imshow(patch_img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis("off")
            
            plt.axis('off')
            plt.savefig(f'{outdir}/{name}_{label}_patches_p{p}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def train_model(
    dataset: Path,
    outdir: Path,
    network: str = 'aovift',
    distribution: str = '/',
    embedding: str = 'spatial_planes',
    samplelimit: Optional[int] = None,
    max_amplitude: float = 1,
    input_shape: int = 64,
    batch_size: int = 32,
    hidden_size: int = 768,
    patches: list = [32, 16],
    heads: list = [8, 8],
    repeats: list = [4, 4],
    depth_scalar: float = 1.,
    width_scalar: float = 1.,
    activation: str = 'gelu',
    fixedlr: bool = False,
    opt: str = 'lamb',
    lr: float = 1e-3,
    wd: float = 1e-2,
    dropout: float = .1,
    warmup: int = 2,
    epochs: int = 5,
    wavelength: float = .510,
    psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
    x_voxel_size: float = .125,
    y_voxel_size: float = .125,
    z_voxel_size: float = .2,
    modes: int = 15,
    pmodes: Optional[int] = None,
    min_photons: int = 1,
    max_photons: int = 1000000,
    roi: Any = None,
    refractive_index: float = 1.33,
    no_phase: bool = False,
    plot_patchfiy: bool = False,
    lls_defocus: bool = False,
    defocus_only: bool = False,
    radial_encoding_period: int = 16,
    radial_encoding_nth_order: int = 4,
    positional_encoding_scheme: str = 'rotational_symmetry',
    fixed_dropout_depth: bool = False,
    steps_per_epoch: Optional[int] = None,
    stem: bool = False,
    mul: bool = False,
    finetune: Optional[Path] = None,
    cpu_workers: int = -1,
):
    outdir.mkdir(exist_ok=True, parents=True)
    network = network.lower()
    opt = opt.lower()
    restored = False

    if network == 'realspace':
        inputs = (input_shape, input_shape, input_shape, 1)
    else:
        inputs = (3 if no_phase else 6, input_shape, input_shape, 1)

    if defocus_only:  # only predict LLS defocus offset
        pmodes = 1
    elif lls_defocus:  # add LLS defocus offset to predictions
        pmodes = modes + 1 if pmodes is None else pmodes + 1
    else:
        pmodes = modes if pmodes is None else pmodes

    if dataset is None:
        config = dict(
            psf_type=psf_type,
            psf_shape=inputs,
            photons=(min_photons, max_photons),
            n_modes=modes,
            distribution=distribution,
            embedding_option=embedding,
            amplitude_ranges=(-max_amplitude, max_amplitude),
            lam_detection=wavelength,
            batch_size=batch_size,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            refractive_index=refractive_index,
            cpu_workers=cpu_workers,
            lls_defocus=lls_defocus,
            defocus_only=defocus_only,
        )
        train_data = data_utils.create_dataset(config)
    else:
        train_data = data_utils.collect_dataset(
            dataset,
            metadata=False,
            modes=pmodes,
            distribution=distribution,
            embedding=embedding,
            samplelimit=samplelimit,
            max_amplitude=max_amplitude,
            no_phase=no_phase,
            lls_defocus=lls_defocus,
            defocus_only=defocus_only,
            photons_range=(min_photons, max_photons),
            cpu_workers=cpu_workers,
            model_input_shape=inputs
        )

        sample_writer = tf.summary.create_file_writer(f'{outdir}/train_samples/')
        with sample_writer.as_default():
            for s in range(10):
                fig = None
                for i, (img, y) in enumerate(train_data.shuffle(batch_size).take(5)):

                    if plot_patchfiy:
                        if network == 'aovift':
                            plot_multiscale_patches(name=f"{i+(s*5)}", img=img, outdir=outdir, patches=patches)
                        elif network == 'prototype':
                            plot_conv_patches(name=f"{i+(s*5)}", img=img, outdir=outdir, patch_ize=patches[0])
                        else:
                            logger.warning(f"Model {network} does not have a patchfiy layer")

                    img = np.squeeze(img, axis=-1)

                    if fig is None:
                        fig, axes = plt.subplots(5, img.shape[0], figsize=(8, 8))

                    for k in range(img.shape[0]):
                        if k > 2:
                            mphi = axes[i, k].imshow(img[k, :, :], cmap='coolwarm', vmin=-.5, vmax=.5)
                        else:
                            malpha = axes[i, k].imshow(img[k, :, :], cmap='Spectral_r', vmin=0, vmax=2)

                        axes[i, k].axis('off')

                    if img.shape[0] > 3:
                        cax = inset_axes(axes[i, 0], width="10%", height="100%", loc='center left', borderpad=-3)
                        cb = plt.colorbar(malpha, cax=cax)
                        cax.yaxis.set_label_position("left")

                        cax = inset_axes(axes[i, -1], width="10%", height="100%", loc='center right', borderpad=-2)
                        cb = plt.colorbar(mphi, cax=cax)
                        cax.yaxis.set_label_position("right")

                    else:
                        cax = inset_axes(axes[i, -1], width="10%", height="100%", loc='center right', borderpad=-2)
                        cb = plt.colorbar(malpha, cax=cax)
                        cax.yaxis.set_label_position("right")

                tf.summary.image("Training samples", utils.plot_to_image(fig), step=s)

        train_data = train_data.cache()
        train_data = train_data.shuffle(batch_size*10, reshuffle_each_iteration=True)
        train_data = train_data.batch(batch_size)
        train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        steps_per_epoch = tf.data.experimental.cardinality(train_data).numpy()

    if fixedlr:
        def scheduler(epoch, lr): return lr
        logger.info(f"Training steps: [{steps_per_epoch * epochs}]")
    else:
        if warmup > 0:
            warmup_steps = warmup * steps_per_epoch
            decay_steps = (epochs - warmup) * steps_per_epoch
            logger.info(f"Training steps [{epochs}: {steps_per_epoch * epochs}] = "
                        f"({warmup}: {warmup_steps=}) + ({epochs-warmup}: {decay_steps=})")
            
            scheduler = CosineDecay(
                initial_learning_rate=0.,
                decay_steps=epochs - warmup,
                warmup_target=lr,
                warmup_steps=warmup,
                alpha=.01,
            )
        else:
            logger.info(f"Training steps [{epochs}: {steps_per_epoch * epochs}]")
            
            scheduler = CosineDecay(
                initial_learning_rate=lr,
                decay_steps=epochs,
                alpha=.01,
                warmup_target=None,
                warmup_steps=0
            )

    if opt == 'lamb':
        opt = LAMB(learning_rate=lr, weight_decay=wd, beta_1=0.9, beta_2=0.99, clipnorm=1.0)
    elif opt.lower() == 'adamw':
        opt = AdamW(learning_rate=lr, weight_decay=wd, beta_1=0.9, beta_2=0.99)
    else:
        opt = Adam(learning_rate=lr)

    try:  # check if model already exists
        model_path = sorted(outdir.rglob('saved_model.pb'))[::-1][0].parent  # sort models to get the latest checkpoint
        model = load(model_path, model_arch=network)

        if isinstance(model, tf.keras.Model):
            restored = True
            opt = model.optimizer
            training_history = pd.read_csv(outdir / 'logbook.csv', header=0, index_col=0)
            logger.info(f"Training history: {training_history}")

    except Exception as e:
        logger.warning(f"No model found in {outdir}")

    if not restored:  # Build a new model
        if network == 'prototype':
            model = prototype.OpticalTransformer(
                name='Prototype',
                roi=roi,
                stem=stem,
                patches=patches,
                heads=heads,
                repeats=repeats,
                modes=pmodes,
                depth_scalar=depth_scalar,
                width_scalar=width_scalar,
                dropout_rate=dropout,
                activation=activation,
                mul=mul,
                no_phase=no_phase,
                positional_encoding_scheme=positional_encoding_scheme,
                radial_encoding_period=radial_encoding_period,
                radial_encoding_nth_order=radial_encoding_nth_order,
                fixed_dropout_depth=fixed_dropout_depth,
            )
        elif network == 'vit':
            model = vit.VIT(
                name='ViT',
                hidden_size=hidden_size,
                roi=roi,
                stem=stem,
                patches=patches,
                heads=heads,
                repeats=repeats,
                modes=pmodes,
                depth_scalar=depth_scalar,
                width_scalar=width_scalar,
                dropout_rate=dropout,
                activation=activation,
                mul=mul,
                no_phase=no_phase,
                positional_encoding_scheme=positional_encoding_scheme,
                fixed_dropout_depth=fixed_dropout_depth,
            )

        elif network == 'aovift':
            model = aovift.OpticalTransformer(
                name='AOViFT',
                roi=roi,
                stem=stem,
                patches=patches,
                heads=heads,
                repeats=repeats,
                modes=pmodes,
                depth_scalar=depth_scalar,
                width_scalar=width_scalar,
                dropout_rate=dropout,
                activation=activation,
                mul=mul,
                no_phase=no_phase,
                positional_encoding_scheme=positional_encoding_scheme,
                radial_encoding_period=radial_encoding_period,
                radial_encoding_nth_order=radial_encoding_nth_order,
                fixed_dropout_depth=fixed_dropout_depth,
            )

        elif network == 'convnext':
            model = convnext.ConvNext(
                name='ConvNext',
                modes=pmodes,
                repeats=repeats,
                projections=heads,
            )

        elif network == 'otfnet':
            model = otfnet.OTFNet(
                name='OTFNet',
                modes=pmodes
            )

        else:
            raise Exception(f'Network "{network}" is unknown.')

    if restored and finetune is None:
        logger.info(f"Continue training {model.name} restored from {model_path} using {opt.get_config()}")
    else:
        if finetune is not None:
            model = load(finetune, model_arch=network)
            logger.info(model.summary(line_length=125, expand_nested=True))
            logger.info(f"Finetuning {model.name}; {opt.get_config()}")

        else:  # creating a new model
            model = model.build(input_shape=inputs)
            logger.info(model.summary(line_length=125, expand_nested=True))
            logger.info(f"Creating a new model; {opt.get_config()})")

        model.compile(
            optimizer=opt,
            loss=tf.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM),
            metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae', 'mse'],
        )

    tblogger = CSVLogger(
        f"{outdir}/logbook.csv",
        append=True,
    )

    pb_checkpoints = ModelCheckpoint(
        filepath=str(outdir/"tf"),
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    h5_checkpoints = ModelCheckpoint(
        filepath=str(outdir/"keras"/f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.h5"),
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    earlystopping = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=50,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    defibrillator = Defibrillator(
        monitor='loss',
        patience=25,
        verbose=1,
    )

    tensorboard = TensorBoardCallback(
        log_dir=outdir,
        histogram_freq=1,
        profile_batch=100000000
    )
    
    lrScheduler = LearningRateScheduler(scheduler, verbose=1)
    
    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        shuffle=True,
        callbacks=[
            tblogger,
            tensorboard,
            pb_checkpoints,
            h5_checkpoints,
            earlystopping,
            defibrillator,
            lrScheduler,
        ],
    )


def eval_model(
    dataset: Path,
    network: str = 'aovift',
    distribution: str = '/',
    embedding: str = 'spatial_planes',
    samplelimit: Optional[int] = None,
    max_amplitude: float = 1,
    input_shape: int = 64,
    batch_size: int = 32,
    wavelength: float = .510,
    psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
    x_voxel_size: float = .125,
    y_voxel_size: float = .125,
    z_voxel_size: float = .2,
    modes: int = 15,
    min_photons: int = 1,
    max_photons: int = 1000000,
    refractive_index: float = 1.33,
    no_phase: bool = False,
    lls_defocus: bool = False,
):
    if network == 'convnext':
        inputs = (input_shape, input_shape, input_shape, 1)
    else:
        inputs = (3 if no_phase else 6, input_shape, input_shape, 1)

    if dataset is None:
        config = dict(
            psf_type=psf_type,
            psf_shape=inputs,
            photons=(min_photons, max_photons),
            n_modes=modes,
            distribution=distribution,
            embedding_option=embedding,
            amplitude_ranges=(-max_amplitude, max_amplitude),
            lam_detection=wavelength,
            batch_size=batch_size,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            refractive_index=refractive_index,
            cpu_workers=-1
        )
        eval_data = data_utils.create_dataset(config)
    else:
        eval_data = data_utils.collect_dataset(
            dataset,
            metadata=False,
            modes=modes,
            distribution=distribution,
            embedding=embedding,
            samplelimit=samplelimit,
            max_amplitude=max_amplitude,
            no_phase=no_phase,
            lls_defocus=lls_defocus,
            photons_range=(min_photons, max_photons),
            model_input_shape=inputs
        )

        eval_data = eval_data.cache()
        eval_data = eval_data.batch(batch_size)
        eval_data = eval_data.prefetch(buffer_size=tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        eval_data = eval_data.with_options(options)

    model = load(network)
    results = model.evaluate(
        eval_data,
        verbose=1,
    )


def parse_args(args):
    train_parser = cli.argparser()

    train_parser.add_argument(
        "--network", default='aovift', type=str, help="codename for target network to train"
    )

    train_parser.add_argument(
        "--dataset", type=Path, help="path to dataset directory"
    )

    train_parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save trained models'
    )

    train_parser.add_argument(
        "--batch_size", default=2048, type=int, help="number of images per batch"
    )
    
    train_parser.add_argument(
        "--hidden_size", default=768, type=int, help="hidden size of transformer block"
    )

    train_parser.add_argument(
        "--patches", default='32-16-8-8', help="patch size for transformer-based model"
    )
    
    train_parser.add_argument(
        "--heads", default='2-4-8-16', help="patch size for transformer-based model"
    )
        
    train_parser.add_argument(
        "--repeats", default='2-4-6-2', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--roi", default=None, help="region of interest to crop from the center of the input image"
    )

    train_parser.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        help="type of the desired PSF"
    )

    train_parser.add_argument(
        "--x_voxel_size", default=.125, type=float, help='lateral voxel size in microns for X'
    )

    train_parser.add_argument(
        "--y_voxel_size", default=.125, type=float, help='lateral voxel size in microns for Y'
    )

    train_parser.add_argument(
        "--z_voxel_size", default=.2, type=float, help='axial voxel size in microns for Z'
    )

    train_parser.add_argument(
        "--input_shape", default=64, type=int, help="PSF input shape"
    )

    train_parser.add_argument(
        "--modes", default=15, type=int, help="number of modes to describe aberration"
    )

    train_parser.add_argument(
        "--pmodes", default=None, type=int, help="number of modes to predict"
    )

    train_parser.add_argument(
        "--min_photons", default=1, type=int, help="minimum photons for training samples"
    )

    train_parser.add_argument(
        "--max_photons", default=10000000, type=int, help="maximum photons for training samples"
    )

    train_parser.add_argument(
        "--dist", default='/', type=str, help="distribution of the zernike amplitudes"
    )

    train_parser.add_argument(
        "--embedding", default='spatial_planes', type=str, help="embedding option to use for training"
    )

    train_parser.add_argument(
        "--samplelimit", default=None, type=int, help="max number of files to load from a dataset [per bin/class]"
    )

    train_parser.add_argument(
        "--max_amplitude", default=1., type=float, help="max amplitude for the zernike coefficients"
    )

    train_parser.add_argument(
        "--wavelength", default=.510, type=float, help='wavelength in microns'
    )

    train_parser.add_argument(
        "--depth_scalar", default=1., type=float, help='scale the number of blocks in the network'
    )

    train_parser.add_argument(
        "--width_scalar", default=1., type=float, help='scale the number of channels in each block'
    )

    train_parser.add_argument(
        '--fixedlr', action='store_true',
        help='toggle to use a fixed learning rate'
    )

    train_parser.add_argument(
        '--mul', action='store_true',
        help='toggle to multiply ratio (alpha) and phase (phi) in the STEM block'
    )

    train_parser.add_argument(
        "--lr", default=1e-3, type=float,
        help='initial learning rate; optimal config: 1e-3 for LAMB and 5e-4 for AdamW'
    )

    train_parser.add_argument(
        "--wd", default=1e-2, type=float, help='initial weight decay; optimal config: 1e-2 for LAMB and 5e-6 for AdamW'
    )

    train_parser.add_argument(
        "--dropout", default=0.1, type=float, help='initial dropout rate for stochastic depth'
    )

    train_parser.add_argument(
        "--opt", default='lamb', type=str, help='optimizer to use for training'
    )

    train_parser.add_argument(
        "--activation", default='gelu', type=str, help='activation function for the model'
    )

    train_parser.add_argument(
        "--warmup", default=25, type=int, help='number of epochs for the initial linear warmup'
    )

    train_parser.add_argument(
        "--epochs", default=500, type=int, help="number of training epochs"
    )

    train_parser.add_argument(
        "--steps_per_epoch", default=100, type=int, help="number of steps per epoch"
    )

    train_parser.add_argument(
        "--cpu_workers", default=8, type=int, help='number of CPU cores to use'
    )

    train_parser.add_argument(
        "--gpu_workers", default=-1, type=int, help='number of GPUs to use'
    )

    train_parser.add_argument(
        '--multinode', action='store_true',
        help='toggle for multi-node/multi-gpu training on a slurm-based cluster'
    )

    train_parser.add_argument(
        '--no_phase', action='store_true',
        help='toggle to use exclude phase from the model embeddings'
    )

    train_parser.add_argument(
        '--lls_defocus', action='store_true',
        help='toggle to also predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--defocus_only', action='store_true',
        help='toggle to only predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--positional_encoding_scheme', default='rotational_symmetry', type=str,
        help='toggle to use different radial encoding types/schemes'
    )

    train_parser.add_argument(
        '--radial_encoding_period', default=16, type=int,
        help='toggle to add more periods for each sin/cos layer in the radial encodings'
    )

    train_parser.add_argument(
        '--radial_encoding_nth_order', default=4, type=int,
        help='toggle to define the max nth zernike order in the radial encodings'
    )

    train_parser.add_argument(
        '--stem', action='store_true',
        help='toggle to use a stem block'
    )

    train_parser.add_argument(
        '--fixed_dropout_depth', action='store_true',
        help='toggle to linearly increase dropout rate for deeper layers'
    )

    train_parser.add_argument(
        "--fixed_precision", action='store_true',
        help='optional toggle to disable automatic mixed precision training'
             '(https://www.tensorflow.org/guide/mixed_precision)'
    )

    train_parser.add_argument(
        "--eval", action='store_true',
        help='evaluate on validation set'
    )

    train_parser.add_argument(
        "--finetune", default=None, type=Path,
        help='evaluate on validation set'
    )

    return train_parser.parse_known_args(args)[0]


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)

def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.multinode:
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
        )
    else:
        strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logger.info(f'Number of active GPUs: {gpu_workers}')

    """
        To enable automatic mixed precision training for tensorflow:
        https://www.tensorflow.org/guide/mixed_precision
        https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#mptrain
        https://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf
    """

    if not args.fixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    with strategy.scope():
        with options({"layout_optimizer": False}):
            
            if args.eval:
                eval_model(
                    dataset=args.dataset,
                    network=args.network,
                    embedding=args.embedding,
                    input_shape=args.input_shape,
                    batch_size=args.batch_size,
                    psf_type=args.psf_type,
                    x_voxel_size=args.x_voxel_size,
                    y_voxel_size=args.y_voxel_size,
                    z_voxel_size=args.z_voxel_size,
                    modes=args.modes,
                    min_photons=args.min_photons,
                    max_photons=args.max_photons,
                    max_amplitude=args.max_amplitude,
                    distribution=args.dist,
                    samplelimit=args.samplelimit,
                    wavelength=args.wavelength,
                    no_phase=args.no_phase,
                    lls_defocus=args.lls_defocus,
                )
            else:
                train_model(
                    dataset=args.dataset,
                    embedding=args.embedding,
                    outdir=args.outdir,
                    network=args.network,
                    input_shape=args.input_shape,
                    batch_size=args.batch_size,
                    hidden_size=args.hidden_size,
                    patches=[int(i) for i in args.patches.split('-')],
                    heads=[int(i) for i in args.heads.split('-')],
                    repeats=[int(i) for i in args.repeats.split('-')],
                    roi=[int(i) for i in args.roi.split('-')] if args.roi is not None else args.roi,
                    steps_per_epoch=args.steps_per_epoch,
                    psf_type=args.psf_type,
                    x_voxel_size=args.x_voxel_size,
                    y_voxel_size=args.y_voxel_size,
                    z_voxel_size=args.z_voxel_size,
                    modes=args.modes,
                    activation=args.activation,
                    mul=args.mul,
                    opt=args.opt,
                    lr=args.lr,
                    wd=args.wd,
                    dropout=args.dropout,
                    fixedlr=args.fixedlr,
                    warmup=args.warmup,
                    epochs=args.epochs,
                    pmodes=args.pmodes,
                    min_photons=args.min_photons,
                    max_photons=args.max_photons,
                    max_amplitude=args.max_amplitude,
                    distribution=args.dist,
                    samplelimit=args.samplelimit,
                    wavelength=args.wavelength,
                    depth_scalar=args.depth_scalar,
                    width_scalar=args.width_scalar,
                    no_phase=args.no_phase,
                    lls_defocus=args.lls_defocus,
                    defocus_only=args.defocus_only,
                    radial_encoding_period=args.radial_encoding_period,
                    radial_encoding_nth_order=args.radial_encoding_nth_order,
                    positional_encoding_scheme=args.positional_encoding_scheme,
                    stem=args.stem,
                    fixed_dropout_depth=args.fixed_dropout_depth,
                    finetune=args.finetune,
                    cpu_workers=args.cpu_workers
                )

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
