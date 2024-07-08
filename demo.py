import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from PIL import Image

import memory_management
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from lib_layerdiffuse.utils import ResizeMode, crop_and_resize_image, rgba2rgbfp32
from utility import LayerMethod, load_models

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
) = load_models(
    "./models",
    "./models/juggernautXL_version6Rundiffusion.safetensors",
    LayerMethod.BG_TO_BLEND.value,
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

positive_tags = "man in room"
default_negative = "face asymmetry, eyes asymmetry, deformed eyes, open mouth"
height = 144
width = 112

with torch.inference_mode():
    guidance_scale = 7.0

    rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)

    bg_image = Image.open("./results/bg.png")
    # assert(bg.mode == "RGBA")
    if bg_image.mode == "RGBA":
        bg_image = rgba2rgbfp32(np.array(bg_image))
    else:
        bg_image = np.array(bg_image, dtype=np.float32) / 255.0
    bg_image = crop_and_resize_image(bg_image, ResizeMode.CROP_AND_RESIZE, height * 8, width * 8) if bg_image is not None else None
    if bg_image is not None:
        bg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(bg_image[..., None].transpose((3, 2, 0, 1)).copy()))).latent_dist.sample()
        # bg_image = unet.model.latent_format.process_in(bg_image)
        bg_image *= 0.13025
    cond = bg_image

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

    memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    result_list, vis_list = transparent_decoder(vae, latents)

    os.makedirs("./results", exist_ok=True)
    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f"./results/bg2ble_{i}_transparent.png", format="PNG")
