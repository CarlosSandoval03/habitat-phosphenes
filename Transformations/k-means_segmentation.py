# K-means segmentation of Gibson virtual environments

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io, measure

def main():
    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())

    path_obs = str(Path("~/home/carlos/Documents/Radboud University/Internship/habitat-phosphenes/EnvironmentExamples/"
                        "episode=109/ezgif-frame-001.jpg").expanduser())

    observation = io.imread(path_obs)

    cv2.imshow(observation)


if __name__ == '__main__':
    main()

    sys.exit()