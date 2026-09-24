import base64
import math
import os
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image


API_URL = "https://api.openai.com/v1/images"


def tensor_to_pil(tensor):
    if tensor is None or tensor.nelement() == 0:
        return None
    array = np.clip(tensor[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3], "RGB")


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, ...])


def image_file(image, name):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return name, buffer.getvalue(), "image/png"


def match_input_size(image):
    width, height = image.size
    ratio = width / height
    if ratio > 3:
        width = height * 3
    elif ratio < 1 / 3:
        height = width * 3

    scale = max(1.0, math.sqrt(655360 / (width * height)))
    scale = min(scale, 3840 / width, 3840 / height, math.sqrt(8294400 / (width * height)))
    width = max(16, round(width * scale / 16) * 16)
    height = max(16, round(height * scale / 16) * 16)
    return f"{width}x{height}"


class GPTImageEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Professional photo edit", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-2"],),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "size": (["match_input", "auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"],),
                "quality": (["auto", "low", "medium", "high", "xhigh", "max"],),
                "background": (["auto", "opaque", "transparent"],),
                "output_format": (["png", "jpeg", "webp"],),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "references": ("GPT_IMAGE_REFS", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "process"
    CATEGORY = "GPT Image"

    def process(self, prompt, negative_prompt, model, api_key, seed, size, quality, background, output_format, output_compression, num_images, image=None, references=None):
        del seed  # GPT Image does not expose deterministic seeding; this input controls ComfyUI re-execution.
        api_key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key missing. Set api_key or OPENAI_API_KEY.")

        primary = tensor_to_pil(image)
        reference_images = [tensor_to_pil(ref) for ref in (references or [])]
        reference_images = [ref for ref in reference_images if ref is not None]
        input_images = ([primary] if primary is not None else []) + reference_images

        if not prompt.strip():
            prompt = "Edit the first image using the other images as references." if input_images else "Create an image."
        if negative_prompt.strip():
            prompt = f"{prompt.rstrip()}\n\nAvoid: {negative_prompt.strip()}"

        requested_size = size
        if size == "match_input":
            requested_size = match_input_size(primary) if primary is not None else "auto"

        request_data = {
            "model": model,
            "prompt": prompt,
            "n": num_images,
            "size": requested_size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
        if output_format in ("jpeg", "webp"):
            request_data["output_compression"] = output_compression

        headers = {"Authorization": f"Bearer {api_key}"}
        if input_images:
            files = [("image[]", image_file(img, f"image_{index}.png")) for index, img in enumerate(input_images)]
            fields = {key: str(value) for key, value in request_data.items()}
            response = requests.post(f"{API_URL}/edits", headers=headers, data=fields, files=files, timeout=600)
            mode = "edit"
        else:
            response = requests.post(f"{API_URL}/generations", headers={**headers, "Content-Type": "application/json"}, json=request_data, timeout=600)
            mode = "generation"

        if not response.ok:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"OpenAI Images API HTTP {response.status_code}: {detail}")

        payload = response.json()
        outputs = []
        for item in payload.get("data", []):
            encoded = item.get("b64_json")
            if encoded:
                outputs.append(pil_to_tensor(Image.open(BytesIO(base64.b64decode(encoded)))))

        if not outputs:
            raise RuntimeError("OpenAI Images API returned no image data.")

        if size == "match_input" and primary is not None:
            restored = []
            for tensor in outputs:
                output = tensor_to_pil(tensor)
                if output.size != primary.size:
                    output = output.resize(primary.size, Image.Resampling.LANCZOS)
                restored.append(pil_to_tensor(output))
            outputs = restored

        usage = payload.get("usage") or {}
        log = (
            f"GPT Image {mode} completed\n"
            f"Model: {model}\n"
            f"Images: {len(outputs)}\n"
            f"Inputs: {len(input_images)} (primary={primary is not None}, references={len(reference_images)})\n"
            f"Requested size: {requested_size}\n"
            f"Quality: {quality}\n"
            f"Usage: {usage or 'not returned'}"
        )
        return torch.cat(outputs, dim=0), log



