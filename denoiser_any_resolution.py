import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
import yaml
import math

class SUNetDenoiser:
    def __init__(self, model_weights, config_path='training.yaml', patch_size=256, stride=256, batch_size=8):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(BASE_DIR, "training.yaml")
        with open(config_path, "r") as f:
            opt =  yaml.safe_load(f)
        
        self.model = SUNet_model(opt).cuda()
        self.load_checkpoint(model_weights)
        self.model.eval()
        
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
    
    def load_checkpoint(self, weights):
        checkpoint = torch.load(weights, map_location='cuda')
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
    
    def overlapped_square(self, timg, kernel, stride):
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
    
    def denoise_frame(self, frame_bgr):
        """
        Denoise a single frame (BGR format from OpenCV)
        Returns: denoised frame in BGR format
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        input_ = TF.to_tensor(frame_rgb).unsqueeze(0).cuda()
        
        with torch.no_grad():
            patches, mask, max_wh = self.overlapped_square(
                input_, kernel=self.patch_size, stride=self.stride
            )
            
            num_patches = patches.shape[0]
            output_patches = torch.zeros_like(patches)
            
            # Process patches in batches
            for i in range(0, num_patches, self.batch_size):
                end_idx = min(i + self.batch_size, num_patches)
                batch = patches[i:end_idx].squeeze(1)
                output_patches[i:end_idx] = self.model(batch).unsqueeze(1)
            
            # Reshape for folding
            B, _, C, PH, PW = output_patches.shape
            output_patches = output_patches.squeeze(1)
            
            # Prepare for fold
            patch_folded = output_patches.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
            patch_folded = patch_folded.permute(2, 1, 3, 0)
            patch_folded = patch_folded.contiguous().view(1, C * self.patch_size * self.patch_size, -1)
            
            # Weight mask
            weight = torch.ones(B, C, PH, PW, device=output_patches.device, dtype=output_patches.dtype)
            weight_mask = weight.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
            weight_mask = weight_mask.permute(2, 1, 3, 0)
            weight_mask = weight_mask.contiguous().view(1, C * self.patch_size * self.patch_size, -1)
            
            # Reconstruct
            restored = F.fold(patch_folded, output_size=(max_wh, max_wh), 
                            kernel_size=self.patch_size, stride=self.stride)
            we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), 
                          kernel_size=self.patch_size, stride=self.stride)
            restored = restored / we_mk
            
            # Crop to original size
            restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
            restored = torch.clamp(restored, 0, 1)
        
        # Convert back to numpy BGR
        restored = restored.permute(0, 2, 3, 1).cpu().numpy()
        restored = img_as_ubyte(restored[0])
        restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        
        return restored_bgr