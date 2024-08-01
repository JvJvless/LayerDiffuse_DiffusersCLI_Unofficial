import os
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace

import safetensors.torch as sf
import torch
import yaml
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from diffusers.utils import is_accelerate_available
from torch.hub import download_url_to_file
from transformers import CLIPTextModel, CLIPTokenizer

from layerdiffuse.lib_layerdiffuse.vae import (
    TransparentVAEDecoder,
    TransparentVAEEncoder,
)

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

hf_endpoint = os.environ["HF_ENDPOINT"] if "HF_ENDPOINT" in os.environ else "https://huggingface.co"


class LayerMethod(Enum):
    def _info(filename: str, description: str, unet_input_channels: int) -> SimpleNamespace:
        return SimpleNamespace(filename=filename, description=description, unet_input_channels=unet_input_channels)

    FG_ONLY_ATTN = _info(filename="layer_xl_transparent_attn.safetensors", description="(SDXL) Only Generate Transparent Image (Attention Injection)", unet_input_channels=4)
    FG_ONLY_CONV = _info(filename="layer_xl_transparent_conv.safetensors", description="(SDXL) Only Generate Transparent Image (Conv Injection)", unet_input_channels=4)
    FG_TO_BLEND = _info(filename="layer_xl_fg2ble.safetensors", description="(SDXL) From Foreground to Blending", unet_input_channels=8)
    FG_BLEND_TO_BG = _info(filename="layer_xl_fgble2bg.safetensors", description="(SDXL) From Foreground and Blending to Background", unet_input_channels=12)
    BG_TO_BLEND = _info(filename="layer_xl_bg2ble.safetensors", description="(SDXL) From Background to Blending", unet_input_channels=8)
    BG_BLEND_TO_FG = _info(filename="layer_xl_bgble2fg.safetensors", description="(SDXL) From Background and Blending to Foreground", unet_input_channels=12)


def download_sth(path: str, url: str):
    if not os.path.exists(path):
        if url is None:
            raise ValueError("No file or url when get state_dict.")
        temp_path = path + ".tmp"
        download_url_to_file(url=url, dst=temp_path)
        os.rename(temp_path, path)

    return path


def load_models(model_dir: str, ckpt_path: str, method: LayerMethod):
    """
    Only SDXL models Supported

    - model_dir (str)
    - ckpt_path (str)
        - must be a safetensors file (SDXL model from CivitAi.com)
    - method (Enum)

    Return:
    - tokenizer
    - tokenizer_2
    - text_encoder
    - text_encoder_2
    - vae
    - origin_unet
    - unet
    - transparent_encoder
    - transparent_decoder
    """
    origin_st = sf.load_file(download_sth(ckpt_path, None))
    modified_st = origin_st.copy()
    inject_st = sf.load_file(
        download_sth(
            os.path.join(model_dir, method.value.filename),
            f"{hf_endpoint}/LayerDiffusion/layerdiffusion-v1/resolve/main/{method.value.filename}",
        )
    )
    with open(
        download_sth(
            os.path.join(model_dir, "sd_xl_base.yaml"),
            "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
        ),
        "r",
    ) as f:
        config = yaml.safe_load(f.read())

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        pad_token="!",
    )
    text_encoder = convert_ldm_clip_checkpoint(
        origin_st,
    )
    text_encoder_2 = convert_open_clip_checkpoint(
        origin_st,
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        prefix="conditioner.embedders.1.model.",
        has_projection=True,
        projection_dim=1280,
    )
    transparent_encoder = TransparentVAEEncoder(
        download_sth(
            os.path.join(model_dir, "vae_transparent_encoder.safetensors"),
            f"{hf_endpoint}/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors",
        )
    )
    transparent_decoder = TransparentVAEDecoder(
        download_sth(
            os.path.join(model_dir, "vae_transparent_decoder.safetensors"),
            f"{hf_endpoint}/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
        )
    )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext

    # Build VAE
    vae_config = create_vae_diffusers_config(config, image_size=1024)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(origin_st, vae_config)
    if "model" in config and "params" in config["model"] and "scale_factor" in config["model"]["params"]:
        vae_scaling_factor = config["model"]["params"]["scale_factor"]
    else:
        vae_scaling_factor = 0.18215  # default SD scaling factor
    vae_config["scaling_factor"] = vae_scaling_factor
    with ctx():
        vae = AutoencoderKL(**vae_config)
    if is_accelerate_available():
        for param_name, param in converted_vae_checkpoint.items():
            set_module_tensor_to_device(vae, param_name, "cpu", value=param)
    else:
        vae.load_state_dict(converted_vae_checkpoint)

    # Build origin UNet
    unet_config = create_unet_diffusers_config(config, image_size=1024)
    unet_config["upcast_attention"] = None

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        origin_st,
        unet_config,
        path="",
        extract_ema=False,
    )
    with ctx():
        origin_unet = UNet2DConditionModel(**unet_config)
    if is_accelerate_available():
        for param_name, param in converted_unet_checkpoint.items():
            set_module_tensor_to_device(origin_unet, param_name, "cpu", value=param)
    else:
        origin_unet.load_state_dict(converted_unet_checkpoint)

    # Build modified UNet
    unet_config["in_channels"] = method.value.unet_input_channels
    for it in inject_st.keys():
        key, inject_type, i = it.split("::")
        key = "model." + key

        if inject_type == "lora":
            if i == "0":
                modified_st[key] += (inject_st[it].to(torch.float32) @ inject_st[f"{it[:-1]}1"].to(torch.float32)).to(torch.float16)
        elif inject_type == "diff":
            w1 = modified_st[key]
            w2 = inject_st[it]
            if w1.shape != w2.shape:
                if w1.ndim == w2.ndim == 4:
                    new_shape = [max(n, m) for n, m in zip(w1.shape, w2.shape)]
                    print(f"[INFO] Merged with {key} channel changed to {new_shape}")
                    weight = torch.zeros(size=new_shape).to(w1)
                    weight[: w1.shape[0], : w1.shape[1], : w1.shape[2], : w1.shape[3]] = w1
                    weight[: w2.shape[0], : w2.shape[1], : w2.shape[2], : w2.shape[3]] += w2
                    w1 = weight.contiguous().clone()
                else:
                    print("[WARNING] SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
            else:
                w1 += w2
            modified_st[key] = w1

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        modified_st,
        unet_config,
        path="",
        extract_ema=False,
    )
    with ctx():
        unet = UNet2DConditionModel(**unet_config)
    if is_accelerate_available():
        for param_name, param in converted_unet_checkpoint.items():
            set_module_tensor_to_device(unet, param_name, "cpu", value=param)
    else:
        unet.load_state_dict(converted_unet_checkpoint)

    return (
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        vae,
        origin_unet,
        unet,
        transparent_encoder,
        transparent_decoder,
    )
