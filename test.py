from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)

download_from_original_stable_diffusion_ckpt("./models/realisticVisionV51_v51VAE.safetensors", from_safetensors=True)
