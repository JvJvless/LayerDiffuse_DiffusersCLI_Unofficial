import os

from torch.hub import download_url_to_file

hf_endpoint = os.environ["HF_ENDPOINT"] if "HF_ENDPOINT" in os.environ else "https://huggingface.co"


def download_sth(path: str, url: str = None):
    if path is None or not os.path.exists(path):
        if url is None:
            raise ValueError("No file or url when get state_dict.")
        temp_path = path + ".tmp"
        download_url_to_file(url=url, dst=temp_path)
        os.rename(temp_path, path)

    return path
