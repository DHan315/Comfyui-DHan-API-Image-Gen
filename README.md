# Comfyui-DHan-API-Image-gen

Gemini Nano Banana and OpenAI GPT Image generation and editing in one ComfyUI node.

Both providers are included. Clone `https://github.com/DHan315/Comfyui-DHan-API-Image-gen` into `ComfyUI/custom_nodes` and restart ComfyUI.

## Nodes

- **DHan-API-Image-gen** lets you choose the provider and model from dropdowns and connect the primary image and references directly.
- **DHan-API-Image-gen RefStacker** shows 14 reference image sockets, like the former Nano stacker, and collects them into one connection for either provider. Direct reference inputs remain available on the main node. GPT Image accepts up to 16 input images total, including the primary image.

The primary image is sent first, followed by references in stack order. `match_input` requests a valid custom GPT Image size near the primary image's aspect ratio, then restores the exact input dimensions for downstream compositing.

## Authentication

Enter an API key in the node, or set `GOOGLE_API_KEY` for Gemini and `OPENAI_API_KEY` for GPT Image before starting ComfyUI. Do not save or publish workflows containing a key.

Switching providers updates the model and supported settings. Provider-specific values are restored when switching back during the current session; the API key is cleared the first time you switch to a provider without a saved key.

## Notes

- `gpt-image-2.5-sunburst` is the precision-editing default; `gpt-image-2.5-flare` is the faster option.
- GPT Image has no seed parameter. The node's seed exists only to let ComfyUI intentionally re-run a request.
- `xhigh` and `max` are GPT Image 2.5 quality settings. Use `high` or lower with `gpt-image-2`.
- Transparent backgrounds require PNG or WebP.
- API use is billed by OpenAI. The log reports usage when the endpoint returns it; the node does not maintain estimated billing totals.

Official API guide: <https://developers.openai.com/api/docs/guides/image-generation>

