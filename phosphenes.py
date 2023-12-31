import copy
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

import gym
from gym.spaces import Box
import os
import pathlib
import importlib.resources
import yaml

import math
import noise

from PIL import Image
import matplotlib.pyplot as plt

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

from IttiSaliencyMap.pySaliencyMap import pySaliencyMap

# Start Added E2E block
savepath= '/home/carsan/Data/habitatai/images/bin/'
# End Added E2E block

def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)

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


@baseline_registry.register_obs_transformer()
class GrayScale(ObservationTransformer):
    def __init__(self): #, resize_needed):
        super().__init__()
        self.transformed_sensor = 'rgb'

        # self.resize_needed = resize_needed

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

        # plt.imsave(savepath + 'obs_input.png', observation[0, :, :, :], cmap=plt.cm.gray)

        frames = []
        for frame in observation:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Start Added E2E block (is this needed? check rest pipeline)
            # Added if resized needed for e2e pipeline
            frame = cv2.resize(frame, (128,128), interpolation=cv2.INTER_AREA)
            # End Added E2E block

            frames.append(frame)

        observation = torch.as_tensor(np.expand_dims(frames, -1),
                                      device=device)
        # Start Added E2E block
        # observation = (observation.float() / 255.0) # Changing scale from 0-255 to 0-1. Without this line the loss is huge
        # plt.imsave('/home/carsan/Data/habitatai/images/gray_norm_output_float_imsave_cmap.png', observation[0,:,:,0].detach().cpu().numpy(), cmap=plt.cm.gray) #desired
        # End Added E2E block

        return observation

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

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

            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
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
            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
            phosphenes = 255 * phosphenes / (phosphenes.max() or 1)
            frames.append(np.tile(np.expand_dims(phosphenes, -1), 3)) # Working (I know for sure)
            # frames.append(np.expand_dims(phosphenes, -1)) # Seems to also be working?


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


# Start Added E2E block
# def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None): # Also not used, for checking it after having a working model
#     layer = [
#         nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(n_output),
#         nn.LeakyReLU(),
#         resample_out]
#     if resample_out is None:
#         layer.pop()
#     return layer

class ConvLayer(nn.Module):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_input, out_channels=n_output,  kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_output)
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        out = self.leaky(self.bn(self.conv(x)))
        return out

class ConvLayer2(nn.Module):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1):
        super(ConvLayer2, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_input, out_channels=n_output,  kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.swish = nn.SiLU() #nn.Swish() # Try with nn.LeakyReLU() for good reconstruction

        # Weight initialization to prevent vanishing gradients
        init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        out = self.swish(self.conv(x))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.swish = nn.SiLU() #nn.Swish() # # Try with nn.LeakyReLU() for good reconstruction
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)

        # Weight initialization to prevent vanishing gradients
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        out = self.swish(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.swish(out)
        return out

class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=1, out_channels=1,binary_stimulation=True):
        super(E2E_Encoder, self).__init__()

        self.binary_stimulation = binary_stimulation

        self.convlayer1 = ConvLayer2(in_channels,8,3,1,1)
        self.convlayer2 = ConvLayer2(8,16,3,1,1)
        self.maxpool1 = nn.MaxPool2d(2) # Not trained
        self.convlayer3 = ConvLayer2(16,32,3,1,1)
        self.maxpool2 =nn.MaxPool2d(2) # Not trained
        self.res1 = ResidualBlock2(32) # 4 trainable parameters each
        self.res2 = ResidualBlock2(32) # 4 trainable parameters each
        self.res3 = ResidualBlock2(32) # 4 trainable parameters each
        self.res4 = ResidualBlock2(32) # 4 trainable parameters each
        self.convlayer4 =ConvLayer2(32,16,3,1,1)
        self.encconv1 = nn.Conv2d(16,out_channels,3,1,1) #bias true
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        # print('encoder input range', x.min(),x.max(), x.shape)

        # Save input Image
        # Conditional to only save images when first dimension is the number of env and not 512?
        # if x.shape[2] == 128:
            # plt.imsave(savepath + 'enc_input.png', x[0, 0, :, :].detach().cpu().numpy(), cmap=plt.cm.gray)

        x = self.convlayer1(x)
        x = self.maxpool1(self.convlayer2(x))
        x = self.maxpool2(self.convlayer3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.tanh1(self.encconv1(x))

        stimulation = .5*(x+1)
        # print('enc_output',stimulation.shape)
        # plt.imsave(savepath+'enc_output.png', stimulation[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        return stimulation

@baseline_registry.register_obs_transformer()
class Encoder(ObservationTransformer):
    def __init__(self, in_channels=1, out_channels=1,binary_stimulation=True):
        super().__init__()

        self.binary_stimulation = binary_stimulation

        self.transformed_sensor = 'rgb'

        #tophos
        self.model = E2E_Encoder()


    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('Input of encoder observation shape', observation.shape)

        stimulation = self.model.forward(observation.permute(0,3,1,2).float())

        # stimulation = .5*(frame+1)  # Is this important?
        stimulation = stimulation.permute(0,2,3,1)

        return stimulation # Continuous space, not only 0s and 1s

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs): # Prob not needed (check after having a working model)
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

class Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SPV representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='relu'):
        super(Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.convlayer1=ConvLayer2(in_channels,16,3,1,1)
        self.convlayer2=ConvLayer2(16,32,3,1,1)
        self.convlayer3=ConvLayer2(32,64,3,2,1)
        self.res1=ResidualBlock2(64) # 4 trainable parameters
        self.res2=ResidualBlock2(64) # 4 trainable parameters
        self.res3=ResidualBlock2(64) # 4 trainable parameters
        self.res4=ResidualBlock2(64) # 4 trainable parameters
        self.convlayer4=ConvLayer2(64,32,3,1,1)
        self.decconv1=nn.Conv2d(32,out_channels,3,1,1)
        self.activ= self.out_activation

    def forward(self, x):
        # print('Decoder input range', x.min(),x.max())

        # plt.imsave(savepath+'dec_input.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)

        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.decconv1(x)
        x = self.activ(x)

        # plt.imsave(savepath+'dec_output.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        # print('decoder output range', x.min(),x.max())

        return x

@baseline_registry.register_obs_transformer()
class E2E_Decoder(ObservationTransformer):
    def __init__(self, in_channels=1, out_channels=1, out_activation='relu'):
        super().__init__()

        self.transformed_sensor = 'rgb'

        self.model = Decoder()

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        observations['rgb'] = self._transform_obs(observations['rgb'])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        observation_sliced=observation.permute(0,3,1,2)[:,0,:,:].unsqueeze(1)

        # print('OBS_SHAPE',observation.shape, observation_sliced.shape)

        reconstruction = self.model.forward(observation_sliced)
        reconstruction = reconstruction.permute(0,2,3,1)

        return reconstruction

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

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

def perlin_noise_map(seed=0,shape=(256,256),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=shape[0],
                                        repeaty=shape[1],
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out

def get_pMask(size=(256,256),phosphene_density=32,seed=1,
              jitter_amplitude=0., intensity_var=0.,
              dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)

    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)

        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density,
                                      jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)

        intensity = intensity_var*(np.random.rand()-0.5)+1.
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,intensity], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = intensity

    return pMask

@baseline_registry.register_obs_transformer()
class E2E_PhospheneSimulator(ObservationTransformer):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SPV template (256x256)
    2. Uses pMask to sample the phosphene locations from the SPV activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15):
        super().__init__()

        # Phosphene grid
        self.pMask = get_pMask(jitter_amplitude=0,dropout=False,seed=0)
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity

        self.transformed_sensor = 'rgb'


    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter


    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('Sim input range', observation.permute(0,3,1,2).min(),observation.permute(0,3,1,2).max())

        # The sim_input torch.Size([64, 1, 32, 32])
        # The sim_output torch.Size([64, 1, 256, 256])

        # plt.imsave(savepath+'sim_input.png', observation.permute(0,3,1,2)[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        device = observation.device

        phosphenes = self.up(observation.permute(0,3,1,2).float())*self.pMask.cuda(device)
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0))
        phosphene = self.intensity*phosphenes

        # print('Sim output range', phosphene.min(),phosphene.max())
        # plt.imsave(savepath+'sim_output.png', phosphene[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        phosphene = torch.tile(phosphene.permute(0,2,3,1), (1,1,1,3))

        return phosphene

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

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

class CustomLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0):

        """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss)
        or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
        stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
        """
        self.recon_loss_type = recon_loss_type
        # Reconstruction loss
        if recon_loss_type == 'mse':
            self.recon_loss = torch.nn.MSELoss()
            self.target = 'image'
        # elif recon_loss_type == 'vgg':
        #     self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
        #     self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
        #     self.target = 'image'
        # elif recon_loss_type == 'boundary':
        #     loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
        #     self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
        #     self.target = 'label'
        elif recon_loss_type == 'ssim':
            self.recon_loss = ssim
            self.target = 'image'
        elif recon_loss_type == 'hist':
            self.recon_loss = self.calculate_histogram_similarity
            self.target = 'image'

        # Stimulation loss
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2)  #torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0

    def prepare(self, observations):
        # Get tensor from dicionary
        key = 'rgb'
        if key in observations:
            observations = observations[key]
        return observations

    def __call__(self,image,stimulation,phosphenes,reconstruction,validation=False):
        device = image.device #self.prepare(image).device

        # Normalization step here instead of gray transform, since now it would correct loss computations without affecting reconstruction
        image = (image.float() / 255.0)

        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            # target = self.prepare(image)
            target = image
        # elif self.target == 'label':
        #     target = labelx

        # Calculate loss (stimu_loss is none, but we are using kappa = 0 so it does not matter anyway
        loss_stimu = self.stimu_loss(stimulation) if self.stimu_loss is not None else torch.zeros(1,device=device)

        if self.recon_loss_type == "mse":
            loss_recon = self.recon_loss(self.prepare(reconstruction),target) # MSE, element wise mean square error
            loss_recon_return = torch.mean(loss_recon)
        elif self.recon_loss_type == "ssim":
            # Calculate SSIM for each image pair in the batch
            ssim_values = [self.recon_loss(img1.cpu().detach().numpy().squeeze(),
                                           img2.cpu().detach().numpy().squeeze(),
                                           data_range=img1.cpu().detach().numpy().squeeze().max() - img1.cpu().detach().numpy().squeeze().min())
                                           for img1, img2 in zip(reconstruction["rgb"], target)]

            # Compute the average SSIM
            average_ssim = sum(ssim_values) / len(ssim_values)
            loss_recon = 1 - average_ssim
            loss_recon = torch.as_tensor(loss_recon, device=device)
            loss_recon_return = loss_recon
        elif self.recon_loss_type == "hist": # Histogram comparison
            histogram_similarities = []
            for img1, img2 in zip(reconstruction["rgb"], target):
                sim = self.recon_loss(img1.cpu().detach().numpy(), img2.cpu().detach().numpy())
                histogram_similarities.append(sim)
            # Compute the average similarity
            loss_recon = sum(histogram_similarities) / len(histogram_similarities)
            loss_recon = torch.as_tensor(loss_recon, device=device)
            loss_recon_return = loss_recon
        else: # MSE for now just to cover all cases
            loss_recon = self.recon_loss(self.prepare(reconstruction), target)
            loss_recon_return = torch.mean(loss_recon)

        loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu

        return loss_total, loss_recon_return, loss_stimu

    def calculate_histogram_similarity(self, target, reconstruction):
        hist_img1 = cv2.calcHist([target], [0], None, [128], [0, 1])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_img2 = cv2.calcHist([reconstruction], [0], None, [128], [0, 1])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Find the metric value
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        return metric_val


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
                maskedObservation = self.get_backgroundMasked_observation(backgroundMask, frame, self.masking_method)

                processedImg = maskedObservation
            else:
                processedImg = frame

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