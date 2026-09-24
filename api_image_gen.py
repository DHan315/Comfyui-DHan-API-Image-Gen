import importlib.util
import math
import os
import sys

from .openai_images import GPTImageEditor


GEMINI_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]
OPENAI_MODELS = [
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2",
]


def load_nano_editor():
    module_name = "comfyui_nanob_edit_gemini_unified"
    module = sys.modules.get(module_name)
    if module is None:
        local_path = os.path.join(os.path.dirname(__file__), "nano_gemini.py")
        spec = importlib.util.spec_from_file_location(module_name, local_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.NanoBEditGemini()


def gpt_size(aspect_ratio, resolution):
    if aspect_ratio in ("match_input", "auto"):
        return aspect_ratio

    ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
    ratio = min(3.0, max(1 / 3, ratio_width / ratio_height))
    target_pixels = {"1K": 1048576, "2K": 4194304, "4K": 8294400}[resolution]
    height = math.sqrt(target_pixels / ratio)
    width = height * ratio
    scale = min(1.0, 3840 / width, 3840 / height)
    width = max(16, round(width * scale / 16) * 16)
    height = max(16, round(height * scale / 16) * 16)
    return f"{width}x{height}"


class APIImageRefStacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"ref_image_{index}": ("IMAGE", {"forceInput": True})
                for index in range(1, 15)
            },
        }

    RETURN_TYPES = ("API_IMAGE_REFS",)
    RETURN_NAMES = ("references",)
    FUNCTION = "stack"
    CATEGORY = "Image API"

    def stack(self, **kwargs):
        return ([
            kwargs[f"ref_image_{index}"]
            for index in range(1, 15)
            if kwargs.get(f"ref_image_{index}") is not None
        ],)


class APIImageGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["Nano Banana (Gemini)", "GPT Image (OpenAI)"],),
                "model": (GEMINI_MODELS + OPENAI_MODELS,),
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "Professional photo edit", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "aspect_ratio": (["match_input", "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9", "4:1", "1:4", "8:1", "1:8"],),
                "resolution": (["0.5K", "1K", "2K", "4K"],),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "low", "medium", "high", "xhigh", "max"],),
                "background": (["auto", "opaque", "transparent"],),
                "output_format": (["png", "jpeg", "webp"],),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100}),
                "thinking_mode": (["Minimal", "High"],),
                "search_grounding": (["Disabled", "Enabled"],),
                "debug_mode": (["Off", "Summary", "Full Request"],),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "references": ("API_IMAGE_REFS", {"forceInput": True}),
                **{
                    f"ref_image_{index}": ("IMAGE", {"forceInput": True})
                    for index in range(1, 15)
                },
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "process"
    CATEGORY = "Image API"

    def process(self, provider, model, api_key, prompt, negative_prompt, seed, aspect_ratio, resolution, num_images, quality, background, output_format, output_compression, thinking_mode, search_grounding, debug_mode, image=None, references=None, **kwargs):
        references = list(references or []) + [
            kwargs[f"ref_image_{index}"]
            for index in range(1, 15)
            if kwargs.get(f"ref_image_{index}") is not None
        ]

        if provider == "Nano Banana (Gemini)":
            if model not in GEMINI_MODELS:
                model = GEMINI_MODELS[0]
            nano_ratio = aspect_ratio
            if nano_ratio == "auto":
                nano_ratio = "match_input" if image is not None else "1:1"
            result, log = load_nano_editor().process(**{
                "prompt": prompt,
                "negative prompt": negative_prompt,
                "model": model,
                "api key": api_key,
                "seed": seed,
                "aspect ratio": nano_ratio,
                "resolution": resolution,
                "num images": num_images,
                "thinking_mode": thinking_mode,
                "search_grounding": search_grounding,
                "safety_filter": "BLOCK_MEDIUM_AND_ABOVE",
                "debug_mode": debug_mode,
                "reset_billing": "Disabled",
                "image": image,
                "references": references,
            })
            return result, f"Provider: Nano Banana (Gemini)\n{log}"

        if model not in OPENAI_MODELS:
            model = OPENAI_MODELS[0]
        result, log = GPTImageEditor().process(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            api_key=api_key,
            seed=seed,
            size=gpt_size(aspect_ratio, resolution),
            quality=quality,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            num_images=num_images,
            image=image,
            references=references,
        )
        return result, f"Provider: GPT Image (OpenAI)\n{log}"

