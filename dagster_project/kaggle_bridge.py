import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi

_api: KaggleApi | None = None


def get_api() -> KaggleApi:
    global _api
    if _api is None:
        _api = KaggleApi()
        _api.authenticate()
    return _api


def push_training_notebook(notebook_dir: str) -> str:
    api = get_api()
    api.kernels_push(notebook_dir)

    with open(os.path.join(notebook_dir, "kernel-metadata.json")) as f:
        metadata = json.load(f)

    return metadata["id"]


def get_kernel_status(kernel_slug: str) -> str:
    api = get_api()
    response = api.kernels_status(kernel_slug)

    if hasattr(response, "status"):
        return response.status.name.lower()
    return response.get("status", "unknown")


def pull_kernel_output(kernel_slug: str, output_dir: str) -> str:
    api = get_api()
    os.makedirs(output_dir, exist_ok=True)
    api.kernels_output(kernel_slug, path=output_dir)
    return output_dir