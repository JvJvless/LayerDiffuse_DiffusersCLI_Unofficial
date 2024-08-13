import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from PIL import Image

import memory_management
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from layerdiffuse.lib_layerdiffuse.utils import (
    ResizeMode,
    crop_and_resize_image,
    rgba2rgbfp32,
)
from utility import LayerMethod, load_models_sdxl

# 加载模型
(
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    vae,
    origin_unet,
    unet,
    transparent_encoder,
    transparent_decoder,
) = load_models_sdxl(
    "./models",
    "./models/juggernautXL_version6Rundiffusion.safetensors",
    LayerMethod.FG_BLEND_TO_BG,
)

pipe = KDiffusionStableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    origin_unet=origin_unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
    ending_step=1000,
    timesteps=1000,
)


def load_condiction(path: str):
    im = Image.open(path)
    # assert(bg.mode == "RGBA")
    if im.mode == "RGBA":
        im = rgba2rgbfp32(np.array(im))
    else:
        im = np.array(im, dtype=np.float32) / 255.0
    im = crop_and_resize_image(im, ResizeMode.CROP_AND_RESIZE, height * 8, width * 8) if im is not None else None
    if im is not None:
        im = 2.0 * torch.from_numpy(np.ascontiguousarray(im[None]).copy()).movedim(-1, 1) - 1.0
        im = vae.encode(im).latent_dist.sample()
        im *= 0.13025
    return im


positive_tags = "bed room, high quality"
default_negative = "bad, ugly"
height = 144
width = 112

with torch.inference_mode():
    guidance_scale = 7.0

    rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)

    # 对于双条件的method，需要concat起来
    cond_1 = load_condiction("./results/fg.png")
    #     cond = load_condiction("./results/bg.png")

    cond_2 = load_condiction("./results/fg2ble_0_transparent.png")

    cond = torch.cat((cond_1, cond_2), dim=1)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    positive_cond, positive_pooler = pipe.encode_cropped_prompt_77tokens(positive_tags)
    negative_cond, negative_pooler = pipe.encode_cropped_prompt_77tokens(default_negative)

    memory_management.load_models_to_gpu([unet, origin_unet])
    initial_latent = torch.zeros(size=(1, 4, height, width), dtype=unet.dtype, device=unet.device)
    latents = pipe(
        initial_latent=initial_latent,
        cond=cond,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images

    # 2blend 不需要transparent_decode
    # 对于2fg的，改用下两行被注释的代码
    memory_management.load_models_to_gpu([vae, transparent_decoder])
    result_list, vis_list = transparent_decoder(vae, latents)

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    pixels = (pixels * 0.5 + 0.5).clip(0, 1).movedim(1, -1)

    result_list = []
    for i in range(int(pixels.shape[0])):
        ret = (pixels[i] * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
        result_list.append(ret)

    os.makedirs("./results", exist_ok=True)
    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f"./results/fg+blend2bg_{i}_transparent.png", format="PNG")
