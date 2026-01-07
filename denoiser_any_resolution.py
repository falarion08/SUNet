import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
import yaml
import math

def load_yaml_config():
    """Load training.yaml relative to this script file."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "training.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def overlapped_square(timg, kernel=256, stride=128):
    b, c, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 3, X, X, device=timg.device, dtype=timg.dtype)
    mask = torch.zeros(1, 1, X, X, device=timg.device, dtype=timg.dtype)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)
    patch = patch.permute(2, 0, 1, 4, 3)

    return patch, mask, X


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cuda')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def denoise_image(
    image_path,
    model_weights,
    patch_size=256,
    stride=128,
    batch_size=8  # Process multiple patches at once
):
    """
    Optimized denoising for a single image.
    
    Returns:
        restored: np.uint8 array of shape H x W x 3
    """
    # Load config and model
    opt = load_yaml_config()
    
    from model.SUNet import SUNet_model
    model = SUNet_model(opt).cuda()
    load_checkpoint(model, model_weights)
    model.eval()

    # Load image
    img = Image.open(image_path).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        # Get patches (now returns all patches as a single tensor)
        patches, mask, max_wh = overlapped_square(input_, kernel=patch_size, stride=stride)
        
        num_patches = patches.shape[0]
        
        # Pre-allocate output tensor
        output_patches = torch.zeros_like(patches)
        
        # Process patches in batches
        for i in range(0, num_patches, batch_size):
            end_idx = min(i + batch_size, num_patches)
            batch = patches[i:end_idx]
            
            # Reshape batch to (batch_size, C, H, W)
            batch = batch.squeeze(1)  # Remove the extra dimension
            
            # Process batch
            output_patches[i:end_idx] = model(batch).unsqueeze(1)
        
        # Reshape for folding
        B, _, C, PH, PW = output_patches.shape
        output_patches = output_patches.squeeze(1)  # Remove batch dimension from patches
        
        # Prepare for fold operation
        patch_folded = output_patches.contiguous().view(B, C, -1, patch_size * patch_size)
        patch_folded = patch_folded.permute(2, 1, 3, 0)
        patch_folded = patch_folded.contiguous().view(1, C * patch_size * patch_size, -1)
        
        # Weight mask for averaging overlaps
        weight = torch.ones(B, C, PH, PW, device=output_patches.device, dtype=output_patches.dtype)
        weight_mask = weight.contiguous().view(B, C, -1, patch_size * patch_size)
        weight_mask = weight_mask.permute(2, 1, 3, 0)
        weight_mask = weight_mask.contiguous().view(1, C * patch_size * patch_size, -1)
        
        # Reconstruct image
        restored = F.fold(patch_folded, output_size=(max_wh, max_wh), kernel_size=patch_size, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=patch_size, stride=stride)
        restored = restored / we_mk
        
        # Crop to original size
        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
        restored = torch.clamp(restored, 0, 1)
    
    # Convert to numpy
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    restored = img_as_ubyte(restored[0])
    
    return restored