"""
Image Reconstruction code
command: python3 reconstruct.py --config_file "configs/gpu/imagenet_lfqgan_256_L.yaml" --ckpt_path  /mnt/tmp/imagenet_256_L.ckpt --save_dir "./visualize" --version  "1k" --image_num 50 --image_size 256
"""
import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from taming.models.lfqgan import VQModel

import argparse
try:
	import torch_npu
except:
    pass
import torchvision

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
	model = VQModel(**config.model.init_args)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
	x = x.detach().cpu()
	x = torch.clamp(x, -1., 1.)
	x = (x + 1.)/2.
	x = x.permute(1,2,0).numpy()
	x = (255*x).astype(np.uint8)
	x = Image.fromarray(x)
	if not x.mode == "RGB":
		x = x.convert("RGB")
	return x

def psnr(original, reconstructed):
    # Convert from [-1,1] to [0,1] range
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def main(args):
    config_file = args.config_file
    configs = OmegaConf.load(config_file)
    configs.data.init_args.batch_size = args.batch_size # change the batch size
    configs.data.init_args.test.params.config.size = args.image_size #using test to inference
    configs.data.init_args.test.params.config.subset = args.subset #using the specific data for comparsion

    model = load_vqgan_new(configs, args.ckpt_path).to(DEVICE)
    # Count and print parameters for encoder and decoder
    encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    decoder_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"Encoder parameters: {encoder_params:.1f}M")
    print(f"Decoder parameters: {decoder_params:.1f}M")
    import pdb; pdb.set_trace()

    visualize_dir = args.save_dir
    visualize_version = args.version
    visualize_original = os.path.join(visualize_dir, visualize_version, "original_{}".format(args.image_size))
    visualize_rec = os.path.join(visualize_dir, visualize_version, "rec_{}".format(args.image_size))
    if not os.path.exists(visualize_original):
       os.makedirs(visualize_original, exist_ok=True)
    
    if not os.path.exists(visualize_rec):
       os.makedirs(visualize_rec, exist_ok=True)
    
    # dataset = instantiate_from_config(configs.data)
    # dataset.prepare_data()
    # dataset.setup()

    count = 0
    with torch.no_grad():
        # Load and preprocess the image
        image = torchvision.io.read_image("/home/nsml/MambaNeRV/data/bunny/0001.png").float()
        image = (image / 255.0) * 2 - 1  # Normalize to [-1, 1]
        
        # Get image dimensions
        _, H, W = image.shape
        
        patchify = False
        if patchify:
            # Calculate number of patches
            n_h = (H + 639) // 640  # Ceiling division
            n_w = (W + 639) // 640
            
            # Initialize tensor to store reconstructed patches
            reconstructed = torch.zeros_like(image)
            
            # Process each patch
            for i in range(n_h):
                for j in range(n_w):
                    # Extract patch
                    h_start = i * 640
                    w_start = j * 640
                    h_end = min(h_start + 640, H)
                    w_end = min(w_start + 640, W)
                    
                    # Pad if necessary
                    patch = image[:, h_start:h_end, w_start:w_end]
                    if patch.shape[1:] != (640, 640):
                        padded = torch.zeros(3, 640, 640, device=patch.device)
                        padded[:, :patch.shape[1], :patch.shape[2]] = patch
                        patch = padded
                    
                    # Process patch
                    patch = patch.unsqueeze(0).to(DEVICE)  # Add batch dimension
                    
                    if model.use_ema:
                        with model.ema_scope():
                            reconstructed_patch, _, _ = model(patch)
                    else:
                        reconstructed_patch, _, _ = model(patch)
                    
                    # Place reconstructed patch back
                    reconstructed_patch = reconstructed_patch[0]  # Remove batch dimension
                    if h_end - h_start != 640 or w_end - w_start != 640:
                        reconstructed_patch = reconstructed_patch[:, :h_end-h_start, :w_end-w_start]
                    reconstructed[:, h_start:h_end, w_start:w_end] = reconstructed_patch.cpu()
        else:
            reconstructed, _, _ = model(image)

        # Save the full reconstructed image
        reconstructed_image = custom_to_pil(reconstructed)
        reconstructed_image.save("reconstructed_magvit.png")

        psnr_value = psnr(image, reconstructed)
        print(f"PSNR: {psnr_value:.2f} dB")

    
def get_args():
   parser = argparse.ArgumentParser(description="inference parameters")
   parser.add_argument("--config_file", required=True, type=str)
   parser.add_argument("--ckpt_path", required=True, type=str)
   parser.add_argument("--image_size", default=256, type=int)
   parser.add_argument("--batch_size", default=1, type=int) ## inference only using 1 batch size
   parser.add_argument("--image_num", default=50, type=int)
   parser.add_argument("--subset", default=None)
   parser.add_argument("--version", type=str, required=True)
   parser.add_argument("--save_dir", type=str, required=True)

   return parser.parse_args()
  
if __name__ == "__main__":
  args = get_args()
  main(args)