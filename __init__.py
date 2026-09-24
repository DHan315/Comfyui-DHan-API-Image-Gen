from .api_image_gen import APIImageGen, APIImageRefStacker

NODE_CLASS_MAPPINGS = {
    "APIImageGen": APIImageGen,
    "APIImageRefStacker": APIImageRefStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIImageGen": "Comfyui-DHan-API-Image-gen",
    "APIImageRefStacker": "Comfyui-DHan-API-Image-gen RefStacker",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

