import habitat # To avoid the problem with libllvmlite
import os
import sys
from pathlib import Path
# Config management (change parameters from script)
# from omegaconf import OmegaConf
# import yaml

from habitat_baselines.run import execute_exp
# Need this import to register custom transformers.
import phosphenes

def main():
    # DISPLAY =:11.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())

    path_config = str(Path('~/Internship/PyCharm_projects/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_complete.yaml').expanduser())

    _config = phosphenes.get_config(path_config)

    # config_dict = OmegaConf.to_container(_config, resolve=True)
    # path_config2 = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
    #                        'ppo_pointnav_phosphenes_complete.yaml').expanduser())
    # with open(path_config2, "w") as f:
    #     yaml.dump(config_dict, f)

    execute_exp(_config, 'train') #train or eval


if __name__ == '__main__':
    main()

    sys.exit()
