import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import utils
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import math
from tqdm import tqdm
import yaml

def load_yaml_config():
    """Load training.yaml relative to this script file."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "training.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)
    patch = patch.permute(2, 0, 1, 4, 3)

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def denoise_image(
    image_path,          # path to input image
    model_weights,       # path to SUNet weights
    patch_size=256,      # patch size
    stride=128           # stride
):
    """
    Denoise a single image using the original SUNet code.

    Returns:
        restored: np.uint8 array of shape H x W x 3
    """
    # Load YAML config
    opt = load_yaml_config()

    # Load model
    model = SUNet_model(opt).cuda()
    load_checkpoint(model, model_weights)
    model.eval()

    # Load image
    img = Image.open(image_path).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Process patches
    with torch.no_grad():
        square_input_, mask, max_wh = overlapped_square(input_.cuda(), kernel=patch_size, stride=stride)
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
        for i, data in enumerate(square_input_):
            restored_patch = model(square_input_[i])
            if i == 0:
                output_patch += restored_patch
            else:
                output_patch = torch.cat([output_patch, restored_patch], dim=0)

        B, C, PH, PW = output_patch.shape
        weight = torch.ones(B, C, PH, PH).type_as(output_patch)

        patch = output_patch.contiguous().view(B, C, -1, patch_size*patch_size)
        patch = patch.permute(2, 1, 3, 0)
        patch = patch.contiguous().view(1, C*patch_size*patch_size, -1)

        weight_mask = weight.contiguous().view(B, C, -1, patch_size * patch_size)
        weight_mask = weight_mask.permute(2, 1, 3, 0)
        weight_mask = weight_mask.contiguous().view(1, C * patch_size * patch_size, -1)

        restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=patch_size, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=patch_size, stride=stride)
        restored /= we_mk

        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
        restored = torch.clamp(restored, 0, 1)

    # Convert to HWC numpy
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])
    return restored