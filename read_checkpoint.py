import torch
import torchvision.models as models

# Load the pretrained ResNet18 model from torchvision
# pretrained_model = models.resnet18(pretrained=True)

# Load checkpoint weights into the model
checkpointHC = torch.load('/scratch/big/home/carsan/Internship/PyCharm_projects/habitat-phosphenes/HC2021/1_best_checkpoint_050e.pt')

# checkpointHC_navPolicy = torch.load('/scratch/big/home/carsan/Internship/PyCharm_projects/habitat-phosphenes/HC2021/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31_ckpt.94_spl_0.8003.pth')

checkpoint = torch.load('/scratch/big/home/carsan/Data/phosphenes/habitat/checkpoints/train_NPT_DepthRGB_Original_long/latest.pth')


print("END")
