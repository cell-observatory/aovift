
import multiprocessing as mp
import os
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import time
import sys
import pandas as pd
import tensorflow as tf
from pathlib import Path

from multiprocessing import active_children

import cli
import experimental
import experimental_eval
import slurm_utils

from backend import load_sample
from preprocessing import prep_sample
from embeddings import measure_fourier_snr

import warnings
warnings.filterwarnings("ignore")


def parse_args(args):
    parser = cli.argparser()
    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="func"
    )
    subparsers.required = True

    cluster_nodes_idle = subparsers.add_parser("cluster_nodes_idle")
    cluster_nodes_wait_for_idle = subparsers.add_parser("cluster_nodes_wait_for_idle")
    cluster_nodes_wait_for_idle.add_argument("idle_minimum", type=int, default=4,
                                             help='Minimum number of idle nodes to wait for')

    psnr = subparsers.add_parser("psnr")
    psnr.add_argument("input", type=Path, help="path to input .tif file")
    psnr.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    psnr.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )

    fourier_snr = subparsers.add_parser("fourier_snr")
    fourier_snr.add_argument("input", type=Path, help="path to input .tif file")
    fourier_snr.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    fourier_snr.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    fourier_snr.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    preprocessing = subparsers.add_parser("preprocessing")
    preprocessing.add_argument("input", type=Path, help="path to input .tif file")
    preprocessing.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    preprocessing.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    preprocessing.add_argument(
        "--read_noise_bias", default=5, type=int, help='bias offset for camera noise'
    )
    preprocessing.add_argument(
        "--normalize", action='store_true',
        help='a toggle for rescaling the image to the max value'
    )
    preprocessing.add_argument(
        "--remove_background", action='store_true',
        help='a toggle for background subtraction'
    )
    preprocessing.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    preprocessing.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    preprocessing.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    preprocessing.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    preprocessing.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )

    embeddings = subparsers.add_parser("embeddings")
    embeddings.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    embeddings.add_argument("input", type=Path, help="path to input .tif file")
    embeddings.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    embeddings.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    embeddings.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    embeddings.add_argument(
        "--fov_is_small", action='store_true',
        help='a toggle for cropping input image to match the model\'s FOV'
    )
    embeddings.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    embeddings.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    embeddings.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    embeddings.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    embeddings.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    embeddings.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    embeddings.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    embeddings.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )

    detect_rois = subparsers.add_parser("detect_rois")
    detect_rois.add_argument("input", type=Path, help="path to input .tif file")
    detect_rois.add_argument("--psf", default=None, type=Path, help="path to PSF .tif file")
    detect_rois.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    detect_rois.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    detect_rois.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    detect_rois.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    detect_rois.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    denoise = subparsers.add_parser("denoise")
    denoise.add_argument("model", type=Path, help="path to pretrained denoise tensorflow model",
                         default='../pretrained_models/denoise/20231107_simulatedBeads_v3_32_64_64/')
    denoise.add_argument("input", type=Path, help="path to input .tif file")
    denoise.add_argument("--output", type=Path, help="path to denoised output .tif file", default=None)
    denoise.add_argument("--window_size", default='64-64-64', type=str,
                         help='size of the window to denoise around each point of interest')
    denoise.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model')
    denoise.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    denoise.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    denoise.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    
    gaussian_fit = subparsers.add_parser("gaussian_fit")
    gaussian_fit.add_argument("input", type=Path, help="path to input .tif file")
    gaussian_fit.add_argument(
        "--window_size", default='11-11-11', type=str,
        help='size of the window to denoise around each point of interest'
    )
    gaussian_fit.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    gaussian_fit.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    gaussian_fit.add_argument(
        "--h_maxima_threshold", default=None, type=int,
	    help='threshold for detecting peaks (counts)'
    )
    gaussian_fit.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    gaussian_fit.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    gaussian_fit.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    gaussian_fit.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    gaussian_fit.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    predict_sample = subparsers.add_parser("predict_sample")
    predict_sample.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_sample.add_argument("input", type=Path, help="path to input .tif file")
    predict_sample.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_sample.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    predict_sample.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_sample.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    predict_sample.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_sample.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_sample.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict_sample.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_sample.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_sample.add_argument(
        "--confidence_threshold", default=0.02, type=float,
        help='optional threshold to flag unconfident predictions '
             'based on the standard deviations of the predicted amplitudes for all digital rotations (microns)'
    )
    predict_sample.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_sample.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_sample.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_sample.add_argument(
        "--num_predictions", default=1, type=int,
        help="number of predictions per sample to estimate model's confidence"
    )
    predict_sample.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_sample.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_sample.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_sample.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    predict_sample.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    predict_sample.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    predict_sample.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    predict_sample.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    predict_sample.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )
    predict_sample.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    predict_sample.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )
    predict_sample.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )
    
    predict_sample.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )

    predict_large_fov = subparsers.add_parser("predict_large_fov")
    predict_large_fov.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_large_fov.add_argument("input", type=Path, help="path to input .tif file")
    predict_large_fov.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_large_fov.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    predict_large_fov.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_large_fov.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    predict_large_fov.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_large_fov.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_large_fov.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict_large_fov.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_large_fov.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_large_fov.add_argument(
        "--confidence_threshold", default=0.02, type=float,
        help='optional threshold to flag unconfident predictions '
             'based on the standard deviations of the predicted amplitudes for all digital rotations (microns)'
    )
    predict_large_fov.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_large_fov.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_large_fov.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_large_fov.add_argument(
        "--num_predictions", default=1, type=int,
        help="number of predictions per sample to estimate model's confidence"
    )
    predict_large_fov.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_large_fov.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_large_fov.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_large_fov.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    predict_large_fov.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    predict_large_fov.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    predict_large_fov.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    predict_large_fov.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    predict_large_fov.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )
    predict_large_fov.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    predict_large_fov.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )
    predict_large_fov.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )

    predict_large_fov.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )
    predict_large_fov.add_argument(
        "--interpolate_embeddings", action='store_true',
        help="`predict_large_fov` will tile the image into small patches and average the FFTs to make the embeddings, "
             "this toggle will compute the FFT of the entire image then downsample/upsample embeddings to match model's input size"
    )

    predict_rois = subparsers.add_parser("predict_rois")
    predict_rois.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_rois.add_argument("input", type=Path, help="path to input .tif file")
    predict_rois.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_rois.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )

    predict_rois.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_rois.add_argument(
        "--window_size", default='64-64-64', type=str, help='size of the window to crop around each point of interest'
    )
    predict_rois.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_rois.add_argument(
        "--num_rois", default=10, type=int,
        help='max number of detected points to use for estimating aberrations'
    )
    predict_rois.add_argument(
        "--min_intensity", default=100, type=int,
        help='minimum intensity desired for detecting peaks of interest'
    )
    predict_rois.add_argument(
        "--minimum_distance", default=.5, type=float,
        help='minimum distance to the nearest neighbor (microns)'
    )
    predict_rois.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    predict_rois.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_rois.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_rois.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_rois.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_rois.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_rois.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_rois.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_rois.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per ROI to estimate model's confidence"
    )
    predict_rois.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_rois.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for modes you wish to ignore'
    )
    predict_rois.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    predict_rois.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    predict_rois.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    predict_rois.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    predict_rois.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    predict_rois.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )
    predict_rois.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    predict_rois.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )
    predict_rois.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )

    predict_tiles = subparsers.add_parser("predict_tiles")
    predict_tiles.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_tiles.add_argument("input", type=Path, help="path to input .tif file")
    predict_tiles.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_tiles.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )

    predict_tiles.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_tiles.add_argument(
        "--window_size", default='64-64-64', type=str, help='size of the window to crop each tile'
    )
    predict_tiles.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_tiles.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    predict_tiles.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_tiles.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_tiles.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_tiles.add_argument(
        "--confidence_threshold", default=0.015, type=float,
        help='optional threshold to flag unconfident predictions '
             'based on the standard deviations of the predicted amplitudes for all digital rotations (microns)'
    )
    predict_tiles.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_tiles.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )
    predict_tiles.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_tiles.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_tiles.add_argument(
        "--num_predictions", default=1, type=int,
        help="number of predictions per tile to estimate model's confidence"
    )
    predict_tiles.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_tiles.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_tiles.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    predict_tiles.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    predict_tiles.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    predict_tiles.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    predict_tiles.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    predict_tiles.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )
    predict_tiles.add_argument(
        "--shift", default=0, type=int,
        help='optional flag for applying digital x shift'
    )
    predict_tiles.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    predict_tiles.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )
    predict_tiles.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )

    predict_folder = subparsers.add_parser("predict_folder")
    predict_folder.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_folder.add_argument("input", type=Path, help="path to input directory with *.tif files")
    predict_folder.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_folder.add_argument(
        "--filename_pattern",
        default=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif", type=str,
        help="optional regex pattern for selecting files in the given directory (Default: `r'*[!_gt|!_realspace|!_noisefree].tif'`)"
    )
    predict_folder.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    predict_folder.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_folder.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_folder.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    predict_folder.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_folder.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_folder.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_folder.add_argument(
        "--confidence_threshold", default=0.015, type=float,
        help='optional threshold to flag unconfident predictions '
             'based on the standard deviations of the predicted amplitudes for all digital rotations (microns)'
    )
    predict_folder.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_folder.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )
    predict_folder.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_folder.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_folder.add_argument(
        "--num_predictions", default=1, type=int,
        help="number of predictions per tile to estimate model's confidence"
    )
    predict_folder.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_folder.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_folder.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )
    predict_folder.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    predict_folder.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    predict_folder.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    predict_folder.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    predict_folder.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )
    predict_folder.add_argument(
        "--shift", default=0, type=int,
        help='optional flag for applying digital x shift'
    )
    predict_folder.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )
    predict_folder.add_argument(
        "--min_psnr", default=5, type=int,
        help='Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold'
    )
    predict_folder.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )

    aggregate_predictions = subparsers.add_parser("aggregate_predictions")

    aggregate_predictions.add_argument("input", type=Path, help="path to csv file")
    aggregate_predictions.add_argument("dm_calibration", type=Path,
                                       help="path DM calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)")

    aggregate_predictions.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM current_dm .csv file (Default: `blank mirror`)"
    )
    aggregate_predictions.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    aggregate_predictions.add_argument(
        "--prediction_threshold", default=.25, type=float,
        help='set predictions below threshold to zero (p2v waves)'
    )
    aggregate_predictions.add_argument(
        "--majority_threshold", default=.5, type=float,
        help='majority rule to use to determine dominant modes among ROIs'
    )
    aggregate_predictions.add_argument(
        "--aggregation_rule", default='median', type=str,
        help='rule to use to calculate final prediction [mean, median, min, max]'
    )
    aggregate_predictions.add_argument(
        "--min_percentile", default=5, type=int,
        help='minimum percentile to filter out outliers'
    )
    aggregate_predictions.add_argument(
        "--max_percentile", default=95, type=int,
        help='maximum percentile to filter out outliers'
    )
    aggregate_predictions.add_argument(
        "--max_isoplanatic_clusters", default=3, type=int,
        help='maximum number of unique isoplanatic patchs for clustering tiles'
    )
    aggregate_predictions.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    aggregate_predictions.add_argument(
        "--ignore_tile", action='append', default=None,
        help='IDs [e.g., "z0-y0-x0"] for tiles you wish to ignore'
    )
    aggregate_predictions.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    aggregate_predictions.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    aggregate_predictions.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    aggregate_predictions.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    aggregate_predictions.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )

    decon = subparsers.add_parser("decon")
    decon.add_argument("input", type=Path, help="path to csv file")
    decon.add_argument(
        "--iters", default=10, type=int,
        help="number of iterations for Richardson-Lucy deconvolution")
    decon.add_argument(
        "--prediction_threshold", default=.25, type=float,
        help='set predictions below threshold to zero (p2v waves)'
    )
    decon.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    decon.add_argument(
        "--ignore_tile", action='append', default=None,
        help='IDs [e.g., "z0-y0-x0"] for tiles you wish to ignore'
    )
    decon.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    decon.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    decon.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    decon.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    decon.add_argument(
        "--only_use_ideal_psf", action='store_true',
        help='a toggle to run only decon with the ideal psf'
    )
    decon.add_argument(
        "--decon_tile", action='store_true',
        help='a toggle to decon each tile independently'
    )
    decon.add_argument("--task", type=str, help='algorithm : "decon", "cocoa"', default='decon')

    combine_tiles = subparsers.add_parser("combine_tiles")
    combine_tiles.add_argument("input", type=Path, help="path to csv file")
    combine_tiles.add_argument(
        "--corrections", action='append', default=[], type=Path,
        help='paths to corrected scans for each DM'
    )
    combine_tiles.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    combine_tiles.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    combine_tiles.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    phase_retrieval = subparsers.add_parser("phase_retrieval")
    phase_retrieval.add_argument("input", type=Path, help="path to input .tif file")
    phase_retrieval.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    phase_retrieval.add_argument(
        "--num_modes", type=int, default=15,
        help="number of zernike modes to predict"
    )
    phase_retrieval.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    phase_retrieval.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    phase_retrieval.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    phase_retrieval.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    phase_retrieval.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    phase_retrieval.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    phase_retrieval.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    phase_retrieval.add_argument(
        "--num_iterations", default=150, type=int,
        help="max number of iterations"
    )
    phase_retrieval.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )

    phase_retrieval.add_argument(
        "--use_pyotf_zernikes", action='store_true',
        help='a toggle to use pyOTF zernike definitions'
    )
    phase_retrieval.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    phase_retrieval.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    phase_retrieval.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    phase_retrieval.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    eval_dm = subparsers.add_parser("eval_dm")
    eval_dm.add_argument("datadir", type=Path, help="path to dataset directory")
    eval_dm.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    eval_dm.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    eval_dm.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )
    eval_dm.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )

    calibrate_dm = subparsers.add_parser("calibrate_dm")
    calibrate_dm.add_argument("datadir", type=Path, help="path to DM eval directory")
    calibrate_dm.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    calibrate_dm.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    calibrate_dm.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    calibrate_dm.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    calibrate_dm.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    eval_mode = subparsers.add_parser("eval_mode")
    eval_mode.add_argument("model_path", type=Path, help="path to pretrained tensorflow model (.h5)")
    eval_mode.add_argument("input_path", type=Path, help="path to input file (.tif)")
    eval_mode.add_argument("gt_path", type=Path, help="path to ground truth file (.csv)")
    eval_mode.add_argument("prediction_path", type=Path, help="path to model predictions (.csv)")
    eval_mode.add_argument("--prediction_postfix", type=str, default='sample_predictions_zernike_coefficients.csv')
    eval_mode.add_argument("--gt_postfix", type=str, default='ground_truth_zernike_coefficients.csv')
    eval_mode.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    eval_mode.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    eval_mode.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    eval_mode.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    eval_beads_dataset = subparsers.add_parser(
        "eval_beads_dataset",
        help="Evaluate artificially introduced aberrations via the DM"
    )
    eval_beads_dataset.add_argument("datadir", type=Path, help="path to dataset directory")
    eval_beads_dataset.add_argument(
        "--flat", default=None, type=Path,
        help="path to the flat DM acts file. If this is given, then DM surface plots will be made."
    )
    eval_beads_dataset.add_argument("--skip_eval_plots", action='store_true', help="skip generating the _ml_eval.svg files.")
    eval_beads_dataset.add_argument("--precomputed", action='store_true')
    eval_beads_dataset.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    eval_beads_dataset.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    eval_beads_dataset.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    eval_beads_dataset.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )


    eval_cell_dataset = subparsers.add_parser(
        "eval_cell_dataset",
        help="Evaluate artificially introduced aberrations via the DM"
    )
    eval_cell_dataset.add_argument("datadir", type=Path, help="path to dataset directory")
    eval_cell_dataset.add_argument(
        "--flat", default=None, type=Path,
        help="path to the flat DM acts file. If this is given, then DM surface plots will be made."
    )
    eval_cell_dataset.add_argument("--precomputed", action='store_true')
    eval_cell_dataset.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    eval_cell_dataset.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    eval_cell_dataset.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    eval_cell_dataset.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    eval_ao_dataset = subparsers.add_parser(
        "eval_ao_dataset",
        help="Evaluate biologically introduced aberrations"
    )
    eval_ao_dataset.add_argument("datadir", type=Path, help="path to dataset directory")
    eval_ao_dataset.add_argument("--flat", default=None, type=Path, help="path to the flat DM acts file")
    eval_ao_dataset.add_argument("--skip_eval_plots", action='store_true',
                                 help="skip generating the _ml_eval.svg files.")
    eval_ao_dataset.add_argument("--precomputed", action='store_true')
    eval_ao_dataset.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    eval_ao_dataset.add_argument(
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
    )
    eval_ao_dataset.add_argument(
        "--partition", type=str, default='abc_a100',
        help="slurm partition to use on the ABC cluster"
    )
    eval_ao_dataset.add_argument(
        "--docker", action='store_true',
        help='a toggle to run predictions through docker container'
    )

    plot_dataset_mips = subparsers.add_parser(
        "plot_dataset_mips",
        help="Evaluate biologically introduced aberrations"
    )
    plot_dataset_mips.add_argument("datadir", type=Path, help="path to dataset directory")

    eval_bleaching_rate = subparsers.add_parser(
        "eval_bleaching_rate",
        help="Evaluate bleaching rates"
    )
    eval_bleaching_rate.add_argument("datadir", type=Path, help="path to dataset directory")

    plot_bleaching_rate = subparsers.add_parser(
        "plot_bleaching_rate",
        help="Visualize bleaching rates evaluations"
    )
    plot_bleaching_rate.add_argument("datadir", type=Path, help="path to dataset directory")

    return parser.parse_known_args(args)


def main(args=None, preloaded=None):
    command_flags = sys.argv[1:] if args is None else args  # raw flags
    args, unknown = parse_args(command_flags)  # parsed flags or passed args
    pd.options.display.width = 200
    pd.options.display.max_columns = 20

    if hasattr(args, 'input'):
        args.input = None if args.input is None else slurm_utils.paths_to_clusterfs(args.input, None)
    if hasattr(args, 'datadir'):
        args.datadir = None if args.datadir is None else slurm_utils.paths_to_clusterfs(args.datadir, None)
    if hasattr(args, 'dm_calibration'):
        args.dm_calibration = None if args.dm_calibration is None else slurm_utils.paths_to_clusterfs(args.dm_calibration, None)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('')
    if not args.docker:
        logger.info(args)

    if os.name != 'nt' and not Path('/clusterfs').exists():
        mount_clusterfs = (r"sudo mkdir /clusterfs && sudo chmod a+wrx /clusterfs/ && "     # make empty directory
                           r"sudo chown 1000:1000 -R /sshkey/ && "  # make /sshkeys (was mounted from host) avail to user 1000
                           r"sshfs thayeralshaabi@login.abc.berkeley.edu:/clusterfs /clusterfs -oIdentityFile=/sshkey/id_rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null ") # sshfs mount without user input
        subprocess.run(mount_clusterfs, shell=True)
        logger.info("Listing filesystem mounts that we just sshfs mounted at /clusterfs (abc, fiona, nvme, nvme2, vast, ...)")
        subprocess.run('ls /clusterfs', shell=True)

    if args.func == 'cluster_nodes_idle':
        number_of_idle_nodes = slurm_utils.get_number_of_idle_nodes(args.partition)
        return number_of_idle_nodes

    if args.func == 'cluster_nodes_wait_for_idle':
        number_of_idle_nodes = 0
        while number_of_idle_nodes < args.idle_minimum:
            if number_of_idle_nodes < args.idle_minimum:
                time.sleep(1)  # Sleep for 1 second
            number_of_idle_nodes = slurm_utils.get_number_of_idle_nodes(args.partition)
            logger.info(f'Number of idle nodes is {number_of_idle_nodes} on {args.partition}. Need {args.idle_minimum}')

        return number_of_idle_nodes

    if args.cluster:
        slurm_utils.submit_slurm_job(args, command_flags, partition=args.partition)

    elif args.docker:
        return slurm_utils.submit_docker_job(args, command_flags, )

    else:
        if os.environ.get('SLURM_JOB_ID') is not None:
            logger.info(f"SLURM_JOB_ID = {os.environ.get('SLURM_JOB_ID')}")
        if os.environ.get('SLURMD_NODENAME') is not None:
            logger.info(f"SLURMD_NODENAME = {os.environ.get('SLURMD_NODENAME')}")
        if os.environ.get('SLURM_JOB_PARTITION') is not None:
            logger.info(f"SLURM_JOB_PARTITION = {os.environ.get('SLURM_JOB_PARTITION')}")

        if os.name == 'nt':
            mp.set_executable(subprocess.run("where python", capture_output=True).stdout.decode('utf-8').split()[0])

        timeit = time.time()

        tf.keras.backend.set_floatx('float32')
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tf.config.experimental.set_memory_growth(gpu_instance, True)

        strategy = tf.distribute.MirroredStrategy(
            devices=[f"{physical_devices[i].device_type}:{i}" for i in range(len(physical_devices))]
        )

        gpu_workers = strategy.num_replicas_in_sync

        if gpu_workers > 0:  # update batchsize automatically
            gpu_model = tf.config.experimental.get_device_details(physical_devices[0])['device_name']
            if gpu_model.find('A100') >= 0:
                args.batch_size = 896 * gpu_workers
            elif gpu_model.find('NVIDIA RTX 6000 Ada Generation') >= 0:
                args.batch_size = 512 * gpu_workers
            elif gpu_model.find('Quadro RTX 8000') >= 0:
                args.batch_size = 512 * gpu_workers
            elif gpu_model.find('NVIDIA TITAN V') >= 0:
                args.batch_size = 128 * gpu_workers
            else:
                pass  # keep custom batch batch size
        else:
            gpu_model = None

        if hasattr(args, 'batch_size'):
            logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}, batch_size={args.batch_size}')

        with strategy.scope():
            if args.func == 'psnr':
                sample = load_sample(args.input)
                prep_sample(
                    sample,
                    remove_background=True,
                    return_psnr=True,
                    plot=None,
                    normalize=False,
                    min_psnr=0,
                    remove_background_noise_method='dog'
                )

            elif args.func == 'fourier_snr':
                sample = load_sample(args.input)
                psnr = prep_sample(
                    sample,
                    remove_background=True,
                    return_psnr=True,
                    plot=None,
                    normalize=False,
                    min_psnr=0,
                    remove_background_noise_method='dog'
                )
                measure_fourier_snr(sample, psnr=psnr, plot=args.input.with_suffix('.svg'))

            elif args.func == 'preprocessing':
                sample_voxel_size = (args.axial_voxel_size, args.lateral_voxel_size, args.lateral_voxel_size)
                sample = load_sample(args.input)
                prep_sample(
                    sample,
                    sample_voxel_size=sample_voxel_size,
                    remove_background=args.remove_background,
                    read_noise_bias=args.read_noise_bias,
                    normalize=args.normalize,
                    plot=args.input.with_suffix('') if args.plot else None,
                    min_psnr=args.min_psnr,
                    remove_background_noise_method='dog'
                )

            elif args.func == 'embeddings':
                experimental.generate_embeddings(
                    file=args.input,
                    model=args.model,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    wavelength=args.wavelength,
                    plot=args.plot,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    preloaded=preloaded,
                    psf_type=args.psf_type,
                    min_psnr=args.min_psnr
                )

            elif args.func == 'predict_sample':
                experimental.predict_sample(
                    model=args.model,
                    img=args.input,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    prev=args.prev,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    wavelength=args.wavelength,
                    dm_damping_scalar=args.dm_damping_scalar,
                    freq_strength_threshold=args.freq_strength_threshold,
                    prediction_threshold=args.prediction_threshold,
                    confidence_threshold=args.confidence_threshold,
                    sign_threshold=args.sign_threshold,
                    num_predictions=args.num_predictions,
                    plot=args.plot,
                    plot_rotations=args.plot_rotations,
                    batch_size=args.batch_size,
                    estimate_sign_with_decon=args.estimate_sign_with_decon,
                    ignore_modes=args.ignore_mode,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    digital_rotations=args.digital_rotations,
                    cpu_workers=args.cpu_workers,
                    preloaded=preloaded,
                    psf_type=args.psf_type,
                    min_psnr=args.min_psnr,
                    estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                    denoiser=args.denoiser
                )

            elif args.func == 'denoise':
                experimental.denoise(
                    input_path=args.input,
                    output_path=args.output,
                    model_path=args.model,
                    window_size=tuple(int(i) for i in args.window_size.split('-')),
                    batch_size=args.batch_size,
                )
            
            elif args.func == 'gaussian_fit':
                experimental.gaussian_fit(
                    img=args.input,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
	                h_maxima_threshold=args.h_maxima_threshold,
                    plot=args.plot,
                    cpu_workers=args.cpu_workers,
                    window_size=tuple(int(i) for i in args.window_size.split('-')),
                )

            elif args.func == 'predict_large_fov':
                experimental.predict_large_fov(
                    model=args.model,
                    img=args.input,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    prev=args.prev,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    wavelength=args.wavelength,
                    dm_damping_scalar=args.dm_damping_scalar,
                    freq_strength_threshold=args.freq_strength_threshold,
                    prediction_threshold=args.prediction_threshold,
                    confidence_threshold=args.confidence_threshold,
                    sign_threshold=args.sign_threshold,
                    num_predictions=args.num_predictions,
                    plot=args.plot,
                    plot_rotations=args.plot_rotations,
                    batch_size=args.batch_size,
                    estimate_sign_with_decon=args.estimate_sign_with_decon,
                    ignore_modes=args.ignore_mode,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    digital_rotations=args.digital_rotations,
                    cpu_workers=args.cpu_workers,
                    preloaded=preloaded,
                    psf_type=args.psf_type,
                    min_psnr=args.min_psnr,
                    estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                    denoiser=args.denoiser,
                    interpolate_embeddings=args.interpolate_embeddings
                )

            elif args.func == 'predict_rois':
                experimental.predict_rois(
                    model=args.model,
                    img=args.input,
                    prev=args.prev,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    wavelength=args.wavelength,
                    window_size=tuple(int(i) for i in args.window_size.split('-')),
                    num_predictions=args.num_predictions,
                    num_rois=args.num_rois,
                    min_intensity=args.min_intensity,
                    freq_strength_threshold=args.freq_strength_threshold,
                    prediction_threshold=args.prediction_threshold,
                    sign_threshold=args.sign_threshold,
                    minimum_distance=args.minimum_distance,
                    plot=args.plot,
                    plot_rotations=args.plot_rotations,
                    batch_size=args.batch_size,
                    estimate_sign_with_decon=args.estimate_sign_with_decon,
                    ignore_modes=args.ignore_mode,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    digital_rotations=args.digital_rotations,
                    cpu_workers=args.cpu_workers,
                    preloaded=preloaded,
                    psf_type=args.psf_type,
                    denoiser=args.denoiser,
                    estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                )
            elif args.func == 'predict_tiles':
                experimental.predict_tiles(
                    model=args.model,
                    img=args.input,
                    prev=args.prev,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    freq_strength_threshold=args.freq_strength_threshold,
                    confidence_threshold=args.confidence_threshold,
                    sign_threshold=args.sign_threshold,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    num_predictions=args.num_predictions,
                    wavelength=args.wavelength,
                    window_size=tuple(int(i) for i in args.window_size.split('-')),
                    plot=args.plot,
                    plot_rotations=args.plot_rotations,
                    batch_size=args.batch_size,
                    estimate_sign_with_decon=args.estimate_sign_with_decon,
                    ignore_modes=args.ignore_mode,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    digital_rotations=args.digital_rotations,
                    cpu_workers=args.cpu_workers,
                    preloaded=preloaded,
                    shifting=(0, 0, args.shift),
                    psf_type=args.psf_type,
                    min_psnr=args.min_psnr,
                    estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                    denoiser=args.denoiser
                )
            elif args.func == 'predict_folder':
                experimental.predict_folder(
                    model=args.model,
                    folder=args.input,
                    filename_pattern=args.filename_pattern,
                    prev=args.prev,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    freq_strength_threshold=args.freq_strength_threshold,
                    confidence_threshold=args.confidence_threshold,
                    sign_threshold=args.sign_threshold,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    num_predictions=args.num_predictions,
                    wavelength=args.wavelength,
                    plot=args.plot,
                    plot_rotations=args.plot_rotations,
                    batch_size=args.batch_size,
                    estimate_sign_with_decon=args.estimate_sign_with_decon,
                    ignore_modes=args.ignore_mode,
                    ideal_empirical_psf=args.ideal_empirical_psf,
                    digital_rotations=args.digital_rotations,
                    cpu_workers=args.cpu_workers,
                    preloaded=preloaded,
                    shifting=(0, 0, args.shift),
                    psf_type=args.psf_type,
                    min_psnr=args.min_psnr,
                    estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                    denoiser=args.denoiser
                )
            elif args.func == 'aggregate_predictions':
                experimental.aggregate_predictions(
                    model_pred=args.input,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    prediction_threshold=args.prediction_threshold,
                    majority_threshold=args.majority_threshold,
                    min_percentile=args.min_percentile,
                    max_percentile=args.max_percentile,
                    aggregation_rule=args.aggregation_rule,
                    max_isoplanatic_clusters=args.max_isoplanatic_clusters,
                    ignore_tile=args.ignore_tile,
                    dm_damping_scalar=args.dm_damping_scalar,
                    plot=args.plot,
                    preloaded=preloaded,
                    psf_type=args.psf_type,
                )
            elif args.func == 'decon':
                experimental.decon(
                    model_pred=args.input,
                    iters=args.iters,
                    ignore_tile=args.ignore_tile,
                    preloaded=preloaded,
                    plot=args.plot,
                    only_use_ideal_psf=args.only_use_ideal_psf,
                    task=args.task,
                    decon_tile=args.decon_tile

                )
            elif args.func == 'combine_tiles':
                experimental.combine_tiles(
                    corrected_actuators_csv=args.input,
                    corrections=args.corrections,
                )
            elif args.func == 'phase_retrieval':
                experimental.phase_retrieval(
                    img=args.input,
                    num_modes=args.num_modes,
                    dm_calibration=args.dm_calibration,
                    dm_state=args.current_dm,
                    axial_voxel_size=args.axial_voxel_size,
                    lateral_voxel_size=args.lateral_voxel_size,
                    wavelength=args.wavelength,
                    dm_damping_scalar=args.dm_damping_scalar,
                    prediction_threshold=args.prediction_threshold,
                    num_iterations=args.num_iterations,
                    plot=args.plot,
                    ignore_modes=args.ignore_mode,
                    use_pyotf_zernikes=args.use_pyotf_zernikes,
                )
            elif args.func == 'eval_dm':
                experimental_eval.eval_dm(
                    datadir=args.datadir,
                )
            elif args.func == 'calibrate_dm':
                experimental_eval.calibrate_dm(
                    datadir=args.datadir,
                    dm_calibration=args.dm_calibration,
                )
            elif args.func == 'eval_mode':
                experimental_eval.eval_mode(
                    model_path=args.model_path,
                    input_path=args.input_path,
                    prediction_path=args.prediction_path,
                    gt_path=args.gt_path,
                    postfix=args.prediction_postfix,
                    gt_postfix=args.gt_postfix,
                )
            elif args.func == 'eval_beads_dataset':
                experimental_eval.eval_beads_dataset(
                    datadir=args.datadir,
                    flat=args.flat,
                    plot_evals=not args.skip_eval_plots,
                    precomputed=args.precomputed,
                )
            elif args.func == 'eval_cell_dataset':
                experimental_eval.eval_cell_dataset(
                    data=args.datadir,
                    flat=args.flat,
                    precomputed=args.precomputed,
                )
            elif args.func == 'eval_ao_dataset':
                experimental_eval.eval_ao_dataset(
                    datadir=args.datadir,
                    flat=args.flat,
                    plot_evals=not args.skip_eval_plots,
                    precomputed=args.precomputed,
                )
            elif args.func == 'plot_dataset_mips':
                experimental_eval.plot_dataset_mips(
                    datadir=args.datadir,
                )
            elif args.func == 'eval_bleaching_rate':
                experimental_eval.eval_bleaching_rate(
                    datadir=args.datadir,
                )
            elif args.func == 'plot_bleaching_rate':
                experimental_eval.plot_bleaching_rate(
                    datadir=args.datadir,
                )
            else:
                logger.error(f"Error")

        logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")

        if hasattr(args, 'input'):
            input = args.input
        elif hasattr(args, 'datadir'):
            input = args.datadir
        else:
            input = None

        if os.name != 'nt' and input is not None:
            logger.info(f"Updating file permissions to {input.parent}")
            subprocess.run(f"find {str(Path(input).parent.resolve())}" + r" -user $USER -exec chmod a+wrx {} +", shell=True)
            subprocess.run(f"find {str(Path(input).parent.resolve())}" + r" -used 46261 -exec chmod a+wrx {} +", shell=True)
            logger.info(f"Updating file permissions complete.")

    return 0


if __name__ == "__main__":
    main()
    # report the number of child processes that are still active
    children = active_children()
    logging.shutdown()
    print(f'Finished. Active children: {len(children)}')
    raise SystemExit(0)
