import habitat # To avoid the problem with libllvmlite
import os
import sys
import torch
from pathlib import Path
# Config management (change parameters from script)
# from omegaconf import OmegaConf
# import yaml

from habitat_baselines.run import execute_exp
# Need this import to register custom transformers.
import phosphenes

def repeat_process():
    path_config = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_test.yaml').expanduser())

    variable_list = [2, 4, 8]
    for variable in variable_list:
        # Load the YAML file into a Python object
        with open(path_config, "r") as f:
            config = yaml.safe_load(f)

        # Modify the value of an existing field in the third embedded dictionary
        config["habitat_baselines"][
            "tensorboard_dir"] = "/home/carsan/Data/phosphenes/habitat/tb/phosphenes" + "/ppo_epoch" + str(variable)
        config["habitat_baselines"]["rl"]["ppo"]["ppo_epoch"] = variable

        # Write the modified Python object back to the YAML file
        with open(path_config, "w") as f:
            yaml.dump(config, f)

        _config = phosphenes.get_config(path_config)

        execute_exp(_config, 'train')

def main():
    # DISPLAY =:11.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Be careful whether I am using 1 or 2 GPUs
    torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(0.9, device=0)
    # memory_summary = torch.cuda.memory_summary(device=None, abbreviated=False)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())

    path_config = str(Path('~/Internship/PyCharm_projects/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_complete.yaml').expanduser())

    _config = phosphenes.get_config(path_config)

    execute_exp(_config, 'train') #train or eval

    # print(memory_summary)


if __name__ == '__main__':
    main()

    sys.exit()
