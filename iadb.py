import torch
import yaml

from diffusers import UNet2DModel
from PIL import Image
from safetensors.torch import save_file, load_file
from torchvision import transforms


#=============================================================================
# Model
#=============================================================================

def config_load(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def config_save(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def model_from_config(config: dict) -> UNet2DModel:
    """Create UNet based on settings from config"""
    channels = config['channels']
    attention = config['attention']
    vae_latents = config['vae_latents']
    levels = len(channels)

    down_blocks = []
    for i in range(levels):
        down_blocks.append("AttnDownBlock2D" if i in attention else "DownBlock2D")
    up_blocks = [x.replace("Down", "Up") for x in reversed(down_blocks)]

    return UNet2DModel(
        block_out_channels=channels,
        out_channels=4 if vae_latents else 3, 
        in_channels=4 if vae_latents else 3, 
        up_block_types=up_blocks, 
        down_block_types=down_blocks, 
        add_attention=True
    )

def model_load_weights(model: UNet2DModel, file_path: str):
    """Loads the model weights from a safetensors file or a training checkpoint"""
    if file_path.endswith('.safetensors'):
        state_dict = load_file(file_path)
    else:
        data = torch.load(file_path, map_location='cpu')
        
        # Check if 'state_dict' is in data
        if isinstance(data, dict) and 'state_dict' in data:
            state_dict = data['state_dict']
        else:
            state_dict = data

        # Re-map layer names
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '')
            new_key = new_key.replace('.query.', '.to_q.')
            new_key = new_key.replace('.key.', '.to_k.')
            new_key = new_key.replace('.value.', '.to_v.')
            new_key = new_key.replace('.proj_attn.', '.to_out.0.')
            cleaned_state_dict[new_key] = v
        state_dict = cleaned_state_dict
    
    # Load state dict
    model.load_state_dict(state_dict)

def model_save_weights(model: UNet2DModel, file_path: str):
    """Saves the model weights to a safetensors file."""
    save_file(model.state_dict(), file_path)

def model_ckpt_to_safetensors(config_path: str, ckpt_path: str, safetensors_path: str):
    """Converts a model checkpoint from training to a safetensors file for inference"""
    config = config_load(config_path)
    model = model_from_config(config)
    model_load_weights(model, ckpt_path)
    save_file(model.state_dict(), safetensors_path)
    del model


#=============================================================================
# Sampling
#=============================================================================

@torch.no_grad()
def inference(model: UNet2DModel, xa: torch.Tensor, a: float) -> torch.Tensor:
    return model(xa, torch.tensor(a, device=xa.device))['sample']

def sample_euler(model: UNet2DModel, x0: torch.Tensor, steps: int) -> torch.Tensor:
    xa, da = x0, 1.0 / steps
    for i in range(steps):
        d = inference(model, xa, i/steps)
        xa += da * d
    return xa

def sample_runge_kutta(model: UNet2DModel, x0: torch.Tensor, steps: int) -> torch.Tensor:
    xa, da = x0, 1.0 / steps
    for i in range(steps):
        d1 = inference(model, xa, i/steps)
        d2 = inference(model, xa + da * d1 / 2, (i+0.5)/steps)
        d3 = inference(model, xa + da * d2 / 2, (i+0.5)/steps)
        d4 = inference(model, xa + da * d3, (i+1.0)/steps)
        xa += da * (d1 + 2*d2 + 2*d3 + d4) / 6
    return xa


#=============================================================================
# Utilities
#=============================================================================

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.squeeze(0).mul(0.5).add(0.5).clamp(0, 1).cpu())
