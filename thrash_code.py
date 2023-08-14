import os
import torch
import sys
import yaml
from pathlib import Path

from habitat_baselines.run import execute_exp
# Need this import to register custom transformers.
import phosphenes

def main():
    # DISPLAY =:11.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    input_path = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/pretrained_models/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth"
    path_config2 = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/pretrained_models/config_from_pretrainedModel"
    state_dict = torch.load(input_path, map_location=torch.device('cpu'))
    config_dict = state_dict["config"]
    with open(path_config2, "w") as f:
        yaml.dump(config_dict, f)

if __name__ == '__main__':
    main()

    sys.exit()