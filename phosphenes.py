import copy
from typing import Dict

import cv2
import numpy as np
from dataclasses import dataclass

import torch
from gym.spaces import Box
import os
import pathlib
import matplotlib.pyplot as plt
import IttiSaliencyMap.pySaliencyMap as pySaliencyMap

from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import (load_params, to_numpy, load_coordinates_from_yaml,
                            Map)
# Dynaphos is a library developed in Neural Coding lab, check paper and git repository.

from habitat import get_config
from habitat.core import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


@baseline_registry.register_obs_transformer()
class GrayScale(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        observation = torch.as_tensor(np.expand_dims(frames, -1),
                                      device=device)

        return observation

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        # c = config.rl.policy.obs_transform.GrayScale
        return cls()


@baseline_registry.register_obs_transformer()
class EdgeFilter(ObservationTransformer):

    def __init__(self, sigma, threshold_low, threshold_high):
        super().__init__()
        self.sigma = sigma
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:

            # Gaussian blur to remove noise.
            frame = cv2.GaussianBlur(frame, ksize=None, sigmaX=self.sigma)

            # Canny edge detection.
            frame = cv2.Canny(frame, self.threshold_low, self.threshold_high)

            # Copy grayscale image on each RGB channel so we can reuse pre-trained net.
            # frames.append(np.tile(np.expand_dims(frame, -1), 3))
            frames.append(np.expand_dims(frame, -1))

        observation = torch.as_tensor(np.array(frames), device=device)

        return observation

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.sigma, config.threshold_low, config.threshold_high)


@baseline_registry.register_obs_transformer()
class Phosphenes(ObservationTransformer):
    def __init__(self, size, phosphene_resolution, sigma):
        super().__init__()
        self.size = size
        self.phosphene_resolution = phosphene_resolution
        self.sigma = sigma
        jitter = 0.4
        intensity_var = 0.8
        aperture = 0.66
        self.transformed_sensor = 'rgb'
        self.grid = create_regular_grid((phosphene_resolution,
                                         phosphene_resolution),
                                        size, jitter, intensity_var)
        # relative aperture > dilation kernel size
        aperture = np.round(aperture *
                            size[0] / phosphene_resolution).astype(int)
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (aperture, aperture))

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.size, config.resolution, config.sigma)

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            mask = cv2.dilate(frame, self.dilation_kernel, iterations=1)
            phosphenes = self.grid * mask
            phosphenes = cv2.GaussianBlur(phosphenes, ksize=None,
                                          sigmaX=self.sigma)

            phosphenes = 255 * phosphenes / (phosphenes.max() or 1)
            frames.append(np.tile(np.expand_dims(phosphenes, -1), 3))

        phosphenes = torch.as_tensor(np.array(frames, 'uint8'), device=device)

        return phosphenes

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 3)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space


@baseline_registry.register_obs_transformer()
class PhosphenesRealistic(ObservationTransformer):
    def __init__(self, phosphene_resolution, intensity_decay, num_electrodes):
        super().__init__()
        self.phosphene_resolution = phosphene_resolution
        self.intensity_decay = intensity_decay
        self.num_electrodes = num_electrodes
        torch.dtype = torch.float32
        self.transformed_sensor = 'rgb'
        self.simulator = self.setup_realistic_phosphenes()

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.resolution, config.intensity_decay, config.num_electrodes)

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def setup_realistic_phosphenes(self):
        # Load the necessary config files from Dynaphos
        path_module = pathlib.Path(__file__).parent.resolve()

        # Feed the configurations from the config file
        params = load_params(os.path.join(path_module, 'dynaphos_files/params.yaml'))
        params['thresholding']['use_threshold'] = False
        params['run']['resolution'] = self.phosphene_resolution
        # params['run']['fps'] = 10
        params['sampling']['filter'] = 'none'

        coordinates_cortex = load_coordinates_from_yaml(os.path.join(path_module, 'dynaphos_files/grid_coords_dipole_valid.yaml'), n_coordinates=self.num_electrodes)
        coordinates_cortex = Map(*coordinates_cortex)
        coordinates_visual_field = get_visual_field_coordinates_from_cortex_full(params['cortex_model'], coordinates_cortex)
        simulator = GaussianSimulator(params, coordinates_visual_field)

        return simulator

    def realistic_representation(self, sensor_observation):
        # Generate phosphenes
        stim_pattern = self.simulator.sample_stimulus(sensor_observation)
        phosphenes = self.simulator(stim_pattern)
        realistic_phosphenes = phosphenes.cpu().numpy() * 255

        return realistic_phosphenes

    def _transform_obs(self, observation):
        device = observation.device
        observation = observation.cpu().numpy()
        frames = []
        for frame in observation:
            frame = np.squeeze(frame)
            phosphenes_realistic = self.realistic_representation(frame)
            # frames.append(np.tile(np.expand_dims(phosphenes_realistic, -1), 3))
            frames.append(np.expand_dims(phosphenes_realistic, -1))

            # Render the image
            # cv2.waitKey(1)
            # cv2.imshow("Edges", phosphenes_realistic)
            # cv2.waitKey(1)

        phosphenes_realistic = torch.as_tensor(np.array(frames, 'uint8'), device=device)

        return phosphenes_realistic


def create_regular_grid(phosphene_resolution, size, jitter, intensity_var):
    """Returns regular eqiodistant phosphene grid of shape <size> with
    resolution <phosphene_resolution> for variable phosphene intensity with
    jittered positions"""

    grid = np.zeros(size)
    phosphene_spacing = np.divide(size, phosphene_resolution)
    xrange = np.linspace(0, size[0], phosphene_resolution[0], endpoint=False) \
        + phosphene_spacing[0] / 2
    yrange = np.linspace(0, size[1], phosphene_resolution[1], endpoint=False) \
        + phosphene_spacing[1] / 2
    for x in xrange:
        for y in yrange:
            deviation = \
                jitter * (2 * np.random.rand(2) - 1) * phosphene_spacing
            intensity = intensity_var * (np.random.rand() - 0.5) + 1
            rx = \
                np.clip(np.round(x + deviation[0]), 0, size[0] - 1).astype(int)
            ry = \
                np.clip(np.round(y + deviation[1]), 0, size[1] - 1).astype(int)
            grid[rx, ry] = intensity
    return grid

@baseline_registry.register_obs_transformer()
class BlackScreen(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        device = observation.device

        observation = observation.cpu().numpy()

        # Create a black screen of the same dimensions as the input frames
        black_screen = np.zeros_like(observation)
        black_screen = np.mean(black_screen, axis=3, keepdims=True)
        observation = torch.as_tensor(black_screen, device=device)

        return observation

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        # c = config.rl.policy.obs_transform.GrayScale
        return cls()


@baseline_registry.register_obs_transformer()
class BackgroundSaliencyDetection(ObservationTransformer):
    def __init__(self, masking_method,background_detection ,saliency_masking):
        super().__init__()
        self.masking_method = masking_method
        self.background_detection = background_detection
        self.saliency_masking = saliency_masking

        if self.saliency_masking:
            self.sm = pySaliencyMap.pySaliencyMap(256, 256)

        self.transformed_sensor = 'rgb'

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.masking_method, config.background_detection, config.saliency_masking)

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            if self.background_detection == True and self.saliency_masking == True:
                backgroundMask = self.get_background(thresholding(gray_scale(gaussian_blur(frame))))
                maskedObservation = self.get_backgroundMasked_observation(backgroundMask, frame, "blur")

                saliency_map = self.sm.SMGetSM(frame)
                saliencyMaskedObs = self.get_saliencyMasked_image(maskedObservation, saliency_map)

                processedImg = saliencyMaskedObs
            elif self.background_detection == False and self.saliency_masking == True:
                saliency_map = self.sm.SMGetSM(frame)
                saliencyMaskedObs = self.get_saliencyMasked_image(frame, saliency_map)

                processedImg = saliencyMaskedObs
            elif self.background_detection == True and self.saliency_masking == False:
                backgroundMask = self.get_background(thresholding(gray_scale(gaussian_blur(frame))))
                maskedObservation = self.get_backgroundMasked_observation(backgroundMask, frame, "blur")

                processedImg = maskedObservation
            else:
                raise Exception("Either background_detection or saliency_masking need to be True for BackgroundSaliencyDetection transformation")

            # frames.append(np.tile(np.expand_dims(maskedObservation, -1), 3))
            frames.append(processedImg)

        observations = torch.as_tensor(np.array(frames, 'uint8'), device=device)

        return observations

    def get_background(self, img, kernelsize=(10, 10)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)
        bin_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
        return sure_bg

    def get_backgroundMasked_observation(self, background_mask, observation, back="blur"):
        background_mask = background_mask / 255
        if back == "black":
            background_mask = background_mask.astype(bool)
            binary_mask_3channel = np.stack((background_mask,) * 3, axis=-1)
            masked_observation = binary_mask_3channel * observation
        elif back == "blur":
            blurred_image = cv2.GaussianBlur(observation, (35, 35), 0)
            foreground_mask = np.stack((background_mask, background_mask, background_mask), axis=-1)
            masked_observation = np.where(foreground_mask, observation, blurred_image)
        else:
            blurred_image = observation
            foreground_mask = np.stack((background_mask, background_mask, background_mask), axis=-1)
            masked_observation = np.where(foreground_mask, observation, blurred_image)

        return masked_observation

    def get_saliencyMasked_image(self, img, saliency_mask):
        # Re-scale saliency mask to get appropriate darkening coefficients
        saliency_normalized = np.zeros_like(saliency_mask)
        saliency_normalized[saliency_mask > 0.3] = 1
        saliency_normalized[(saliency_mask > 0.1) & (saliency_mask <= 0.3)] = 0.6
        saliency_normalized[saliency_mask <= 0.1] = 0.5

        # Darken the image based on the normalized mask
        img_float = img.astype(np.float32) / 255
        darkened_image = img_float * saliency_normalized[:, :, np.newaxis]
        # Convert back to uint8
        darkened_image_uint8 = np.clip(darkened_image * 255, 0, 255).astype(np.uint8)
        return darkened_image_uint8


@baseline_registry.register_obs_transformer()
class SegmentationCV2(ObservationTransformer):
    def __init__(self):
        super().__init__()

        self.transformed_sensor = 'rgb'

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours
                threshold_area = 900
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

                # Draw contours on the original image
                height, width, channels = frame.shape
                zero_image = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(zero_image, filtered_contours, -1, (255, 255, 255), 1)
            else:
                raise Exception("The image observation does not exist.")

            # frames.append(np.tile(np.expand_dims(maskedObservation, -1), 3))
            frames.append(zero_image)

        observations = torch.as_tensor(np.array(frames, 'uint8'), device=device)

        return observations


def gaussian_blur(img, kernel=(5, 5)):
    observation = cv2.GaussianBlur(img, kernel, cv2.BORDER_DEFAULT)
    return observation

def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresholding(img):
    ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img

