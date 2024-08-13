import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionKDiffusionPipeline
from PIL import Image

import memory_management
from layerdiffuse.lib_layerdiffuse.attention_sharing import AttentionSharingPatcher
from utility import LayerMethod, load_models_sd15

prompt = "high quality, 4k"
default_negative = "nsfw, bad, ugly"
fg_additional_prompt = "a man sitting"
bg_additional_prompt = "chair"
blend_additional_prompt = "a man sitting on chair"

height = 640
width = 512

device = "cuda:0"
scheduler = "sample_dpmpp_2m"
guidance_scale = 7.0
seed = 12345

if __name__ == "__main__":
    memory_management.gpu = device

    models = load_models_sd15(
        "./models",
        "./models/realisticVisionV51_v51VAE.safetensors",
        LayerMethod.JOINT_SD15,
    )

    origin_unet = models.pop("origin_unet")
    transparent_encoder = models.pop("transparent_encoder")
    transparent_decoder = models.pop("transparent_decoder")
    patcher: AttentionSharingPatcher = models.pop("patcher")
    vae = models["vae"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]

    pipe = StableDiffusionKDiffusionPipeline(
        scheduler=LMSDiscreteScheduler(),
        requires_safety_checker=False,
        **models,
    )
    pipe.set_scheduler(scheduler)

    def clip_encode(text):
        if text is None:
            return None

        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        cond = text_encoder(tokens.to(device))
        return cond.last_hidden_state

    with torch.inference_mode():

        rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

        fg_additional_prompt = fg_additional_prompt + ", " + prompt if fg_additional_prompt != "" else None
        bg_additional_prompt = bg_additional_prompt + ", " + prompt if bg_additional_prompt != "" else None
        blend_additional_prompt = blend_additional_prompt + ", " + prompt if blend_additional_prompt != "" else None

        # 因为diffusers内置的pipeline的device是遍历modules列表确定的，所以要提前加载好，不然device可能会找错
        memory_management.load_models_to_gpu([text_encoder, unet, vae])
        fg_cond = clip_encode(fg_additional_prompt)
        bg_cond = clip_encode(bg_additional_prompt)
        blend_cond = clip_encode(blend_additional_prompt)

        # memory_management.load_models_to_gpu([text_encoder, unet])
        # pipeline内是 cat[uncond, cond]
        patcher.set_attach_cond(
            {
                "cond_mark": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                "cond_overwrite": [fg_cond, bg_cond, blend_cond],
            }
        )
        latents = pipe(
            prompt=[prompt] * 3,
            height=height,
            width=width,
            num_inference_steps=20,
            guidance_scale=guidance_scale,
            negative_prompt=[default_negative] * 3,
            generator=rng,
            output_type="latent",
            use_karras_sigmas=True,
        ).images

        memory_management.load_models_to_gpu([vae, transparent_decoder])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor

        result_list, _ = transparent_decoder(vae, latents[:1])
        pixels = vae.decode(latents[1:]).sample
        pixels = (pixels * 0.5 + 0.5).clip(0, 1).movedim(1, -1)
        for i in range(int(pixels.shape[0])):
            ret = (pixels[i] * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
            result_list.append(ret)

        os.makedirs("./images", exist_ok=True)
        for i, image in enumerate(result_list):
            Image.fromarray(image).save(f"./images/joint_sd15_{i}_transparent.png", format="PNG")
