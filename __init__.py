from .api_image_gen import APIImageGen

NODE_CLASS_MAPPINGS = {
    "APIImageGen": APIImageGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIImageGen": "API-Image-gen",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

