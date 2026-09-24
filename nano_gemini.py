import os
import torch
import requests
import base64
import json
import concurrent.futures
import shutil
import re
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
from datetime import datetime


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def tensor2pil(t):
    if t is None or (isinstance(t, torch.Tensor) and t.nelement() == 0):
        return None

    i = 255.0 * t[0].cpu().numpy()
    img = np.clip(i, 0, 255).astype(np.uint8)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    return Image.fromarray(img, mode="RGB")


def pil2tensor(p):
    if p is None:
        return None

    arr = np.array(p).astype(np.float32) / 255.0
    return torch.from_numpy(arr[np.newaxis, ...])


def get_closest_ratio(pil_img, model=None):
    if not pil_img:
        return "1:1"

    w, h = pil_img.size
    actual = w / h

    # Use the ratios actually supported by the selected model. Gemini 3.1 Flash
    # supports the extended panoramic ratios; Pro and 2.5 use the base set.
    ratios = {
        "1:1": 1.0,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "21:9": 21 / 9,
    }
    if model == "gemini-3.1-flash-image":
        ratios.update({
            "4:1": 4.0,
            "1:4": 0.25,
            "8:1": 8.0,
            "1:8": 0.125,
        })

    return min(ratios, key=lambda k: abs(ratios[k] - actual))


def get_gemini_native_size(model, ratio, resolution):
    """Return Google's documented native output size for ratio/resolution."""
    flash31_1k = {
        "1:1": (1024, 1024), "1:4": (512, 2048), "1:8": (384, 3072),
        "2:3": (848, 1264), "3:2": (1264, 848), "3:4": (896, 1200),
        "4:1": (2048, 512), "4:3": (1200, 896), "4:5": (928, 1152),
        "5:4": (1152, 928), "8:1": (3072, 384), "9:16": (768, 1376),
        "16:9": (1376, 768), "21:9": (1584, 672),
    }
    flash31_512 = {k: (max(1, w // 2), max(1, h // 2)) for k, (w, h) in flash31_1k.items()}
    # Google's 512 table has a few dimensions that are not exactly half of 1K.
    flash31_512.update({
        "2:3": (424, 632), "3:2": (632, 424), "3:4": (448, 600),
        "4:3": (600, 448), "4:5": (464, 576), "5:4": (576, 464),
        "9:16": (384, 688), "16:9": (688, 384), "21:9": (792, 336),
    })

    pro_1k = {
        k: v for k, v in flash31_1k.items()
        if k in {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
    }
    flash25_1k = {
        "1:1": (1024, 1024), "2:3": (832, 1248), "3:2": (1248, 832),
        "3:4": (864, 1184), "4:3": (1184, 864), "4:5": (896, 1152),
        "5:4": (1152, 896), "9:16": (768, 1344), "16:9": (1344, 768),
        "21:9": (1536, 672),
    }

    if model == "gemini-2.5-flash-image":
        return flash25_1k.get(ratio)

    base = flash31_1k if model == "gemini-3.1-flash-image" else pro_1k
    if resolution == "512":
        return flash31_512.get(ratio) if model == "gemini-3.1-flash-image" else base.get(ratio)
    native = base.get(ratio)
    if native is None:
        return None
    scale = {"1K": 1, "2K": 2, "4K": 4}.get(resolution, 1)
    return native[0] * scale, native[1] * scale


def edge_pad_to_exact_ratio(pil_img, ratio_size):
    """Pad without scaling so the canvas exactly matches ratio_size's reduced ratio."""
    if pil_img is None or not ratio_size:
        return pil_img, (0, 0, 0, 0)

    from math import gcd, ceil

    rw, rh = ratio_size
    g = gcd(int(rw), int(rh))
    unit_w, unit_h = int(rw) // g, int(rh) // g
    w, h = pil_img.size
    multiplier = max(ceil(w / unit_w), ceil(h / unit_h))
    target_w, target_h = unit_w * multiplier, unit_h * multiplier

    pad_w = target_w - w
    pad_h = target_h - h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if not any((left, top, right, bottom)):
        return pil_img, (0, 0, 0, 0)

    arr = np.asarray(pil_img)
    padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")
    return Image.fromarray(padded.astype(np.uint8), mode="RGB"), (left, top, right, bottom)


# ---------------------------------------------------------
# Reference Handling
# ---------------------------------------------------------

class NanoBRefConfig:
    """Legacy compatibility adapter. New workflows should connect IMAGE directly to NanoB Reference Stacker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("NANO_REF_DATA",)
    RETURN_NAMES = ("ref_data",)
    FUNCTION = "make_config"
    CATEGORY = "NanoGemini/Legacy"

    def make_config(self, image):
        return ({"image": image},)


class NanoBRefStacker:
    """Collects up to 14 reference IMAGE inputs. No weighting is applied."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"ref_image_{i}": ("IMAGE", {"forceInput": True})
                for i in range(1, 15)
            },
        }

    RETURN_TYPES = ("NANO_REFS",)
    RETURN_NAMES = ("references",)
    FUNCTION = "stack"
    CATEGORY = "NanoGemini/Reference"

    def stack(self, **kwargs):
        bundle = [
            kwargs.get(f"ref_image_{i}")
            for i in range(1, 15)
            if kwargs.get(f"ref_image_{i}") is not None
        ]
        return (bundle,)


class NanoBEditGemini:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Professional photo edit",
                        "multiline": True,
                    },
                ),
                "negative prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "model": (
                    [
                        "gemini-3.1-flash-image",
                        "gemini-3-pro-image",
                        "gemini-2.5-flash-image",
                    ],
                    {"default": "gemini-3.1-flash-image"},
                ),
                "api key": ("STRING", {"default": ""}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "aspect ratio": (
                    [
                        "match_input",
                        "1:1",
                        "16:9",
                        "9:16",
                        "4:3",
                        "3:4",
                        "2:3",
                        "3:2",
                        "4:5",
                        "5:4",
                        "21:9",
                        "4:1",
                        "1:4",
                        "8:1",
                        "1:8",
                    ],
                    {"default": "match_input"},
                ),
                "resolution": (
                    ["0.5K", "1K", "2K", "4K"],
                    {"default": "1K"},
                ),
                "num images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                    },
                ),
                "thinking_mode": (
                    ["Minimal", "High"],
                    {"default": "Minimal"},
                ),
                "search_grounding": (
                    ["Enabled", "Disabled"],
                    {"default": "Disabled"},
                ),
                "safety_filter": (
                    [
                        "BLOCK_NONE",
                        "BLOCK_ONLY_HIGH",
                        "BLOCK_MEDIUM_AND_ABOVE",
                    ],
                    {"default": "BLOCK_MEDIUM_AND_ABOVE"},
                ),
                "debug_mode": (
                    ["Off", "Summary", "Full Request"],
                    {"default": "Off"},
                ),
                "reset_billing": (
                    ["Disabled", "Reset Now"],
                    {"default": "Disabled"},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "references": ("NANO_REFS", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "log")
    FUNCTION = "process"
    CATEGORY = "NanoGemini"

    def process(self, **kwargs):
        json_path = os.path.join(
            os.path.dirname(__file__),
            "industry_master_billing.json",
        )
        md_path = os.path.join(
            os.path.dirname(__file__),
            "industry_master_billing.md",
        )
        debug_path = os.path.join(
            os.path.dirname(__file__),
            "nano_debug_latest.txt",
        )

        api_log = ""
        debug_mode = kwargs.get("debug_mode", "Off")

        if kwargs.get("reset_billing") == "Reset Now":
            if os.path.exists(json_path):
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                backup_name = f"Billing_Archive_{stamp}.json"
                shutil.copy(
                    json_path,
                    os.path.join(os.path.dirname(__file__), backup_name),
                )
                os.remove(json_path)
                api_log += (
                    f"✅ BILLING RESET: Archive {backup_name} created.\n"
                )

        api_key = (
            kwargs.get("api key", "").strip()
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not api_key:
            raise ValueError("API Key Missing")

        model = kwargs.get("model")
        resolution = kwargs.get("resolution")
        is_pro = "pro" in model

        thinking_mode = kwargs.get("thinking_mode")
        safety_filter = kwargs.get("safety_filter")
        search_grounding = (
            kwargs.get("search_grounding") == "Enabled"
        )

        raw_user_prompt = kwargs.get("prompt", "").strip()
        user_prompt = raw_user_prompt
        negative_prompt = kwargs.get("negative prompt", "").strip()

        # Normalize common shorthand so numbered references are unambiguous.
        user_prompt = re.sub(
            r"\bref(?:erence)?(?:\s+image)?\s*(\d+)\b",
            r"Reference Image \1",
            user_prompt,
            flags=re.IGNORECASE,
        )

        # Primary image is always the first image in the Interactions API input.
        p1 = tensor2pil(kwargs.get("image"))
        orig_size, det_ratio = (
            (p1.size, get_closest_ratio(p1, model))
            if p1
            else (None, "1:1")
        )

        requested_ratio = kwargs.get("aspect ratio")
        match_input_mode = requested_ratio == "match_input" and p1 is not None
        chosen_ratio = det_ratio if requested_ratio == "match_input" else requested_ratio

        # Interactions API uses "512" for the 0.5K size.
        api_resolution = "512" if resolution == "0.5K" else resolution
        if model == "gemini-3-pro-image" and api_resolution == "512":
            api_resolution = "1K"
        if model == "gemini-2.5-flash-image":
            api_resolution = "1K"

        # For match_input, keep the legacy explicit Gemini ratio/resolution request
        # (which produced better edit registration), but pad the source first so
        # the API ratio conversion is deterministic and reversible. No source
        # pixels are scaled or cropped before the request.
        p1_api = p1
        primary_pad = (0, 0, 0, 0)
        primary_padded_size = p1.size if p1 else None
        native_ratio_size = None
        if match_input_mode:
            native_ratio_size = get_gemini_native_size(model, chosen_ratio, api_resolution)
            p1_api, primary_pad = edge_pad_to_exact_ratio(p1, native_ratio_size)
            primary_padded_size = p1_api.size

        bundle = kwargs.get("references", []) or []

        # Build a single-turn, role-ordered multimodal edit request.
        # Donor/reference images are introduced first. The PRIMARY IMAGE is placed
        # immediately before the final edit instruction so the last visual target in
        # the request is the canvas that should be edited, not a donor reference.
        instruction = user_prompt or "Edit the primary image."

        interaction_input = []
        primary_sent_size = None
        reference_sent_sizes = []

        valid_reference_count = 0
        for ref in bundle:
            ref_tensor = ref.get("image") if isinstance(ref, dict) else ref
            img = tensor2pil(ref_tensor)
            if img is None:
                continue

            valid_reference_count += 1
            reference_sent_sizes.append(img.size)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            interaction_input.extend([
                {
                    "type": "text",
                    "text": (
                        f"REFERENCE IMAGE {valid_reference_count} — DONOR REFERENCE ONLY. "
                        "This image supplies appearance/identity traits only when explicitly "
                        "requested. Its pose, head angle, framing, composition, body, clothing, "
                        "background, lighting, and camera are NOT the target and must not be copied."
                    ),
                },
                {
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                },
            ])

        if p1_api:
            primary_sent_size = p1_api.size
            buf = BytesIO()
            p1_api.save(buf, format="JPEG", quality=95)
            pad_note = ""
            if match_input_mode and any(primary_pad):
                pad_note = (
                    " Temporary edge padding has been added outside the original canvas only for "
                    "API aspect-ratio compatibility. Do not reframe, zoom, crop, or shift the inner "
                    "original image because of this padding."
                )
            interaction_input.extend([
                {
                    "type": "text",
                    "text": (
                        "PRIMARY IMAGE — EDIT TARGET / CANVAS. This is the image whose pixels, "
                        "composition, pose, head angle, camera angle, framing, body, clothing, "
                        "background, lighting, and scene must remain the basis of the output. "
                        "Apply the requested donor traits to THIS image and adapt those traits to "
                        "this image's existing geometry and perspective." + pad_note
                    ),
                },
                {
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                },
            ])

        request_text = (
            "EDIT THE PRIMARY IMAGE ABOVE. Do not edit or recreate a reference image. "
            "When transferring a face, hair, identity, material, or other trait from a reference, "
            "adapt that trait to the PRIMARY IMAGE's existing pose, head angle, perspective, "
            "silhouette, composition, and lighting rather than transplanting the reference image's "
            "geometry or replacing the reference with the primary.\n\n"
            "EDIT INSTRUCTION: " + instruction
        ) if p1 else instruction

        if negative_prompt:
            request_text += f"\n\nAVOID: {negative_prompt}"

        interaction_input.append({"type": "text", "text": request_text})

        # Current Interactions API. Explicit ratio/resolution is retained in
        # match_input because this was the more stable edit-registration path.
        url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        response_format = {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": chosen_ratio,
            "image_size": api_resolution,
        }

        payload = {
            "model": model,
            "input": interaction_input,
            "response_format": response_format,
            "generation_config": {
                "seed": int(kwargs.get("seed", 0)),
            },
            "store": False,
        }

        if model == "gemini-3.1-flash-image":
            payload["generation_config"]["thinking_level"] = thinking_mode.lower()

        if search_grounding:
            payload["tools"] = [{
                "type": "google_search",
                "search_types": ["web_search", "image_search"],
            }]

        # NOTE: The public Gemini Interactions API does not accept the
        # `safety_settings` request field used by generateContent. Keep the
        # legacy UI value for saved-workflow compatibility, but intentionally
        # do not include it in the Interactions payload. Gemini API defaults apply.

        # -------------------------------------------------
        # Debug Report (sanitized: no API key or image bytes)
        # -------------------------------------------------

        if debug_mode != "Off":
            debug_lines = [
                "NANOB DEBUG REPORT",
                "=" * 60,
                f"Debug mode: {debug_mode}",
                f"Model: {model}",
                f"Resolution requested: {resolution}",
                f"Resolution sent: {api_resolution}",
                f"Aspect ratio sent: {chosen_ratio}",
                f"Thinking mode: {thinking_mode}",
                f"Search grounding: {search_grounding}",
                f"Safety filter (legacy UI; not sent): {safety_filter}",
                f"Primary image connected: {p1 is not None}",
                f"Primary source size: {orig_size}",
                f"Primary transmitted size: {primary_sent_size}",
                f"Match-input padding L/T/R/B: {primary_pad if match_input_mode else 'n/a'}",
                f"Match-input native ratio size: {native_ratio_size if match_input_mode else 'n/a'}",
                f"Reference count: {valid_reference_count}",
                f"Edit request mode: role-ordered single turn",
                f"Reference transmitted sizes: {reference_sent_sizes}",
                "",
                "RAW USER PROMPT:",
                raw_user_prompt or "[EMPTY]",
                "",
                "NORMALIZED USER PROMPT:",
                user_prompt or "[EMPTY]",
                "",
                "REFERENCE ORDER:",
            ]

            if valid_reference_count:
                for ref_index in range(1, valid_reference_count + 1):
                    debug_lines.append(f"Reference Image {ref_index}")
            else:
                debug_lines.append("[NO REFERENCE IMAGES]")

            if negative_prompt:
                debug_lines.extend([
                    "",
                    "NEGATIVE / AVOID PROMPT:",
                    negative_prompt,
                ])

            if debug_mode == "Full Request":
                debug_lines.extend([
                    "",
                    "SANITIZED MULTIMODAL PARTS IN SEND ORDER:",
                ])

                image_counter = 0
                for part_index, part in enumerate(interaction_input, 1):
                    if part.get("type") == "text":
                        debug_lines.extend([
                            "",
                            f"--- INPUT {part_index}: TEXT ---",
                            part.get("text", ""),
                        ])
                    elif part.get("type") == "image":
                        image_counter += 1
                        mime_type = part.get("mime_type", "unknown")
                        encoded_size = len(part.get("data", ""))
                        debug_lines.extend([
                            "",
                            f"--- INPUT {part_index}: IMAGE {image_counter} ---",
                            f"[IMAGE DATA OMITTED | MIME={mime_type} | BASE64 CHARS={encoded_size}]",
                        ])

                sanitized_config = {
                    "model": payload.get("model"),
                    "tools_enabled": bool(payload.get("tools")),
                    "generation_config": payload.get("generation_config", {}),
                    "response_format": payload.get("response_format", {}),
                    "safety_settings": "Not sent (Interactions API uses Gemini defaults)",
                    "store": payload.get("store"),
                }
                debug_lines.extend([
                    "",
                    "SANITIZED API CONFIG:",
                    json.dumps(sanitized_config, indent=2),
                ])

            debug_text = "\n".join(debug_lines) + "\n"
            api_log += "\n--- DEBUG REPORT ---\n" + debug_text

            try:
                with open(debug_path, "w", encoding="utf-8") as debug_file:
                    debug_file.write(debug_text)
                api_log += f"Debug file saved: {debug_path}\n"
            except Exception as debug_error:
                api_log += f"Debug file save failed: {debug_error}\n"

        def run_api():
            try:
                request_payload = dict(payload)
                request_payload["generation_config"] = dict(payload.get("generation_config", {}))

                response = requests.post(
                    url,
                    headers=headers,
                    json=request_payload,
                    timeout=300,
                )

                if response.status_code == 200:
                    return response.json()

                return {
                    "error": (
                        f"HTTP {response.status_code}: "
                        f"{response.text}"
                    )
                }

            except Exception as exc:
                return {"error": str(exc)}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(run_api)
                for _ in range(kwargs.get("num images"))
            ]
            results = [future.result() for future in futures]

        output_tensors = []

        for result in results:
            if "error" in result:
                api_log += f"API Error: {result['error']}\n"
                continue

            status = result.get("status")
            if status and status not in ("completed", "incomplete"):
                api_log += f"Interaction status: {status}\n"

            # Raw REST Interactions responses expose generated media inside
            # model_output steps (SDK convenience fields like output_image are not
            # guaranteed in REST JSON).
            images_found = []
            for step in result.get("steps", []) or []:
                if step.get("type") != "model_output":
                    continue
                for content in step.get("content", []) or []:
                    if content.get("type") == "image" and content.get("data"):
                        images_found.append(content)
                    elif content.get("type") == "text" and content.get("text"):
                        api_log += f"\nGemini: {content['text']}\n"

            # Defensive compatibility in case the REST service exposes a top-level
            # image helper in a future/variant response.
            top_image = result.get("output_image")
            if isinstance(top_image, dict) and top_image.get("data"):
                images_found.append(top_image)

            for image_content in images_found:
                try:
                    decoded = base64.b64decode(image_content["data"])
                    img_out = Image.open(BytesIO(decoded)).convert("RGB")

                    if match_input_mode and orig_size and primary_padded_size:
                        # Geometry-preserving inverse transform:
                        # 1) uniformly map Gemini's native ratio image back to the
                        #    padded source canvas, then
                        # 2) remove only the temporary padding.
                        # This avoids the non-uniform W/H stretch used in v1.0.4,
                        # which shifted features and broke downstream compositing.
                        if img_out.size != primary_padded_size:
                            api_log += (
                                f"Match-input uniform restore: {img_out.size[0]}x{img_out.size[1]} "
                                f"-> {primary_padded_size[0]}x{primary_padded_size[1]}\n"
                            )
                            img_out = img_out.resize(primary_padded_size, Image.Resampling.LANCZOS)

                        left, top, right, bottom = primary_pad
                        crop_box = (left, top, left + orig_size[0], top + orig_size[1])
                        if crop_box != (0, 0, img_out.size[0], img_out.size[1]):
                            img_out = img_out.crop(crop_box)
                            api_log += (
                                f"Match-input crop restored original canvas: "
                                f"{orig_size[0]}x{orig_size[1]}\n"
                            )

                    output_tensors.append(pil2tensor(img_out))
                except Exception as decode_error:
                    api_log += f"Image decode error: {decode_error}\n"

            if search_grounding:
                usage = result.get("usage", {}) or {}
                grounding_counts = usage.get("grounding_tool_count", []) or []
                if grounding_counts:
                    api_log += "\n🔍 Google Search grounding used.\n"

        if not output_tensors:
            fallback = (
                torch.ones((1, 512, 512, 3))
                * 0.5
            )

            return (
                fallback,
                f"FAILED. Diagnostic:\n{api_log}",
            )

        # -------------------------------------------------
        # Billing Logic
        # -------------------------------------------------

        num_generated = len(output_tensors)

        flash_cost = 0.039
        pro_cost = (
            0.2400
            if resolution == "4K"
            else 0.1344
        )

        current_api_rate = (
            pro_cost
            if is_pro
            else flash_cost
        )

        session_cost = (
            num_generated
            * current_api_rate
        )

        if not is_pro:
            adobe_rate = 0.05
            adobe_credits = 10
            weavy_rate = 0.0036
            weavy_credits = 0.4
        else:
            if resolution == "4K":
                adobe_rate = 1.20
                adobe_credits = 240
                weavy_rate = 0.108
                weavy_credits = 12.0
            else:
                adobe_rate = 0.20
                adobe_credits = 40
                weavy_rate = 0.054
                weavy_credits = 6.0

        if resolution == "1K":
            krea_units = 10
            krea_rate = 0.017
        elif resolution == "2K":
            krea_units = 150
            krea_rate = 0.25
        else:
            krea_units = 300
            krea_rate = 0.50

        if is_pro and resolution == "4K":
            fal_rate = 0.30
        elif is_pro:
            fal_rate = 0.15
        else:
            fal_rate = 0.039

        data = {
            "api_flash": 0.0,
            "api_pro": 0.0,
            "total_api": 0.0,
            "adobe_total": 0.0,
            "adobe_creds_total": 0,
            "fal_total": 0.0,
            "krea_total": 0.0,
            "krea_units_total": 0,
            "weavy_total": 0.0,
            "weavy_creds_total": 0.0,
            "img_total": 0,
        }

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)

                data.update({
                    key: value
                    for key, value in loaded.items()
                    if key in data
                })

            except Exception:
                pass

        if is_pro:
            data["api_pro"] += session_cost
        else:
            data["api_flash"] += session_cost

        data["total_api"] = (
            data["api_flash"]
            + data["api_pro"]
        )

        data["img_total"] += num_generated
        data["adobe_total"] += (
            num_generated * adobe_rate
        )
        data["adobe_creds_total"] += (
            num_generated * adobe_credits
        )
        data["fal_total"] += (
            num_generated * fal_rate
        )
        data["krea_total"] += (
            num_generated * krea_rate
        )
        data["krea_units_total"] += (
            num_generated * krea_units
        )
        data["weavy_total"] += (
            num_generated * weavy_rate
        )
        data["weavy_creds_total"] += (
            num_generated * weavy_credits
        )

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        self.save_markdown_report(md_path, data)

        savings = (
            data["adobe_total"]
            - data["total_api"]
        )

        log = "GEMINI NANO ECONOMIC REPORT\n"
        log += (
            f"Audit: Thinking={thinking_mode} | "
            f"Safety=Gemini API default (legacy UI={safety_filter}) | Debug={debug_mode}\n"
        )
        log += (
            f"References: {valid_reference_count} | "
            "Mode: Explicit labeled edit target + labeled references\n"
        )
        if primary_sent_size:
            log += f"Primary transmitted: {primary_sent_size[0]}x{primary_sent_size[1]}"
            if match_input_mode and any(primary_pad):
                log += f" (temporary geometry-safe padding {primary_pad})"
            else:
                log += " (no geometry transform)"
            log += "\n"
        log += (
            f"Session Cost: ${session_cost:.4f} "
            f"(${current_api_rate:.4f}/img)\n\n"
        )

        log += "LIFETIME API SPEND (REAL SPEND)\n"
        log += (
            f"Total Spend: ${data['total_api']:.2f} "
            f"({data['img_total']} images)\n"
        )
        log += (
            f"Flash Cost:  ${data['api_flash']:.2f} | "
            f"Pro Cost: ${data['api_pro']:.2f}\n\n"
        )

        log += "MARKET BENCHMARK SIMULATION\n"
        log += (
            f"Adobe Firefly: ${data['adobe_total']:.2f} "
            f"({data['adobe_creds_total']:,} credits)\n"
        )
        log += (
            f"Krea Pro:      ${data['krea_total']:.2f} "
            f"({data['krea_units_total']:,} units)\n"
        )
        log += (
            f"Weavy:         ${data['weavy_total']:.2f} "
            f"({data['weavy_creds_total']:.1f} credits)\n"
        )
        log += (
            f"Fal.ai:        ${data['fal_total']:.2f}\n\n"
        )

        log += (
            f"TOTAL SAVINGS VS ADOBE: "
            f"${savings:.2f}\n"
        )
        log += "------------------------------------------\n"
        log += "--- API LOG ---"
        log += api_log

        return (
            torch.cat(output_tensors, dim=0),
            log,
        )

    def save_markdown_report(self, path, data):
        markdown = (
            "# Nano Banana Industry Report\n"
            f"Images: {data['img_total']}\n"
            f"Savings vs Adobe: "
            f"${data['adobe_total'] - data['total_api']:.2f}"
        )

        with open(path, "w", encoding="utf-8") as file:
            file.write(markdown)


NODE_CLASS_MAPPINGS = {
    "NanoBRefConfig": NanoBRefConfig,
    "NanoBRefStacker": NanoBRefStacker,
    "NanoBEditGemini": NanoBEditGemini,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBRefConfig": "NanoB Reference Adapter (Legacy)",
    "NanoBRefStacker": "NanoB Reference Stacker",
    "NanoBEditGemini": "NanoB Edit Gemini",
}
