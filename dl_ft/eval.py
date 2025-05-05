import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchmetrics import SSIM
from torchmetrics.image import PSNR
from torchvision.transforms import ToTensor
from tqdm import tqdm
from omegaconf import OmegaConf

import sys
sys.path.append("./stable_diffusion")

import k_diffusion as K
import einops
from einops import rearrange
from ldm.util import instantiate_from_config

# defined in edit_app.py
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


# define InstructPix2Pix to facilitate evaluation
class InstructPix2Pix(nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str, vae_ckpt: str = None):
        super().__init__()
        # load config
        self.config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(self.config.model)
        
        # load checkpoint
        pl_sd = torch.load(checkpoint_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        
        # load vae checkpoint
        if vae_ckpt is not None:
            vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
            sd = {
                k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
                for k, v in sd.items()
            }
        
        self.model.load_state_dict(sd, strict=False)
        self.model.to("cuda")
        self.model.eval()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""]).to("cuda") 

    @classmethod
    def load_from_checkpoint(cls, config_path: str, checkpoint_path: str, vae_ckpt: str = None):
        return cls(config_path, checkpoint_path, vae_ckpt)
    
    def edit(
        self,
        image: torch.Tensor,
        instruction: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> torch.Tensor:
        """generate image from text"""
        image = image.to("cuda")
        with torch.no_grad(), self.model.ema_scope():
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning([instruction]).to("cuda")],
                "c_concat": [self.model.encode_first_stage(image.unsqueeze(0)).mode().to("cuda")],
            }
            uncond = {
                "c_crossattn": [self.null_token.to("cuda")],
                "c_concat": [torch.zeros_like(cond["c_concat"][0]).to("cuda")],
            }
            
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": guidance_scale,
                "image_cfg_scale": 1.0,
            }
            sigmas = self.model_wrap.get_sigmas(num_inference_steps)
            x = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            
            return self.model.decode_first_stage(x)[0]


def preprocess_image(image: Image.Image, target_size: tuple = (256, 256)) -> Image.Image:
    """resize image to target size"""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def load_model(
    config_path: str,
    checkpoint_path: str,
    vae_ckpt: str = None,
    device: str = "cuda"
) -> InstructPix2Pix:
    """load model"""
    model = InstructPix2Pix.load_from_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        vae_ckpt=vae_ckpt
    )
    return model.to(device)

def generate_prediction(
    model: InstructPix2Pix,
    input_image: Image.Image,
    instruction: str,
    device: str = "cuda",
    steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    """generate prediction"""
    input_tensor = ToTensor()(input_image).to(device)
    output_tensor = model.edit(
        input_tensor,
        instruction=instruction,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    )
    
    # convert to image
    output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    return Image.fromarray(output_image)

def evaluate_metrics(pred_image: Image.Image, target_image: Image.Image, device: str = "cuda") -> dict:
    """compute metrics"""
    ssim = SSIM(data_range=255.0).to(device)
    psnr = PSNR(data_range=255.0).to(device)
    
    pred_tensor = ToTensor()(pred_image).unsqueeze(0).to(device)
    target_tensor = ToTensor()(target_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ssim_score = ssim(pred_tensor, target_tensor)
        psnr_score = psnr(pred_tensor, target_tensor)
    
    return {"SSIM": ssim_score.item(), "PSNR": psnr_score.item()}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        vae_ckpt=args.vae_ckpt,
        device=device
    )
    
    # create output directory
    pred_dir = Path(args.output_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_dir)
    all_metrics = []
    
    print(f"Data root directory: {data_root}")
    if not data_root.exists():
        print(f"ERROR: Data directory {data_root} does not exist!")
        return
    
    # find all view directories
    view_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"Found {len(view_dirs)} view directories: {[d.name for d in view_dirs]}")
    
    total_prompt_dirs = 0
    valid_samples = 0
    
    for view_dir in view_dirs:
        print(f"Processing view: {view_dir.name}")
        # find all prompt directories
        prompt_dirs = [d for d in view_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")]
        print(f"  Found {len(prompt_dirs)} prompt directories in {view_dir.name}")
        total_prompt_dirs += len(prompt_dirs)
        
        for prompt_dir in tqdm(prompt_dirs, desc=f"Processing {view_dir.name} samples"):
            input_path = prompt_dir / "0.jpg"
            target_path = prompt_dir / "1.jpg"
            
            if not input_path.exists() or not target_path.exists():
                print(f"  Warning: Missing input or target files in {prompt_dir}")
                print(f"  Files in directory: {list(prompt_dir.glob('*'))}")
                continue
                
            prompt_json_path = prompt_dir / "prompt.json"
            if not prompt_json_path.exists():
                print(f"  Warning: Missing prompt.json in {prompt_dir}")
                continue
                
            with open(prompt_json_path) as f:
                prompt_data = json.load(f)
                instruction = prompt_data.get("edit", "")
                if not instruction:
                    print(f"  Warning: No 'edit' field in {prompt_json_path}")
                    print(f"  Available fields: {list(prompt_data.keys())}")
                    continue
            
            try:
                input_image = preprocess_image(Image.open(input_path))
                target_image = preprocess_image(Image.open(target_path))
            except Exception as e:
                print(f"  Error opening images in {prompt_dir}: {e}")
                continue
            
            # generate prediction
            try:
                pred_image = generate_prediction(
                    model, input_image, instruction, 
                    steps=args.steps, guidance_scale=args.guidance_scale
                )
                
                # save prediction
                output_path = pred_dir / f"{view_dir.name}_{prompt_dir.name}_pred.jpg"
                pred_image.save(output_path)
                
                # compute metrics
                metrics = evaluate_metrics(pred_image, target_image, device)
                all_metrics.append(metrics)
                valid_samples += 1
                print(f"  Successfully processed {prompt_dir.name}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}")
            except Exception as e:
                print(f"  Error processing {prompt_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nSummary: Found {total_prompt_dirs} prompt directories, successfully processed {valid_samples} samples")
    
    # compute average metrics
    if all_metrics:
        avg_ssim = np.mean([m["SSIM"] for m in all_metrics])
        avg_psnr = np.mean([m["PSNR"] for m in all_metrics])
        
        print("\nEvaluation Results:")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        
        # save results to a file
        results_path = Path(args.output_dir) / "results.txt"
        with open(results_path, "w") as f:
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
    else:
        print("Warning: No valid samples were processed. Check your data structure.")
        

data_dir = "./ft_data"
checkpoint_path = "./logs/train_default/checkpoints/last.ckpt"
config_path = "./logs/train_default/configs/2025-05-04T22-16-01-project.yaml"
output_dir = "./logs/predictions"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help=".ckpt file path")
    parser.add_argument("--config_path", type=str, required=True, help="config file path,eg,./logs/train_default/configs/2025-05-04T22-16-01-project.yaml")
    parser.add_argument("--output_dir", type=str, default="./logs/predictions", help="output directory")

    
    args = parser.parse_args()
    main(args)