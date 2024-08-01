import os

import yaml
from accelerate import init_empty_weights
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from .utils import download_sth


def get_sdxl_config(config_dir: str):
    with open(
        download_sth(
            os.path.join(config_dir, "sd_xl_base.yaml"),
            "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
        ),
        "r",
    ) as f:
        config = yaml.safe_load(f.read())
    return config


def get_sdxl_text_encoder():
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        pad_token="!",
    )

    text_encoder_config = CLIPTextConfig.from_pretrained(
        "openai/clip-vit-large-patch14",
    )
    text_encoder_2_config = CLIPTextConfig.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
    )

    with init_empty_weights():
        text_encoder = CLIPTextModel(text_encoder_config)
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_2_config)

    return tokenizer, tokenizer_2, text_encoder, text_encoder_2


def get_sd15_config(config_dir: str):
    raise NotImplementedError()
