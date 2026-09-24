import { app } from "../../scripts/app.js";

const TARGET = "APIImageGen";
const STACKER = "APIImageRefStacker";
const MAX_REFS = 14;
const MIN_WIDTH = 460;
const STACKER_WIDTH = 240;
const MODELS = {
    "Nano Banana (Gemini)": ["gemini-3.1-flash-image", "gemini-3-pro-image", "gemini-2.5-flash-image"],
    "GPT Image (OpenAI)": ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-2"],
};
const BASE_RATIOS = ["match_input", "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"];
const EXTENDED_RATIOS = ["4:1", "1:4", "8:1", "1:8"];
const GPT_WIDGETS = ["quality", "background", "output_format"];
const GEMINI_WIDGETS = ["debug_mode"];
const PROVIDER_VALUES = [
    "api_key", "model", "aspect_ratio", "resolution", "quality", "background",
    "output_format", "output_compression", "thinking_mode", "search_grounding", "debug_mode",
];

function widget(node, name) {
    return node.widgets?.find(item => item.name === name);
}

function showWidget(item, visible) {
    if (!item) return;
    if (visible && item.__apiImageOriginalType !== undefined) {
        item.type = item.__apiImageOriginalType;
        item.computeSize = item.__apiImageOriginalComputeSize;
        item.hidden = item.__apiImageOriginalHidden;
        delete item.__apiImageOriginalType;
        delete item.__apiImageOriginalComputeSize;
        delete item.__apiImageOriginalHidden;
    } else if (!visible && item.__apiImageOriginalType === undefined) {
        item.__apiImageOriginalType = item.type;
        item.__apiImageOriginalComputeSize = item.computeSize;
        item.__apiImageOriginalHidden = item.hidden;
        item.type = "hidden";
        item.hidden = true;
        item.computeSize = () => [0, -4];
    }
}

function resizeNode(node) {
    const size = node.computeSize?.() || node.size;
    const minWidth = (node.comfyClass || node.type) === TARGET ? MIN_WIDTH : STACKER_WIDTH;
    node.setSize?.([Math.max(node.size?.[0] || 0, minWidth), size[1]]);
}

function setChoices(item, values) {
    if (!item) return;
    item.options.values = values;
    if (!values.includes(item.value)) item.value = values[0];
}

function indexOf(input) {
    const match = /^ref image (\d+)$/.exec(String(input?.name || "").replaceAll("_", " ").toLowerCase());
    return match ? Number(match[1]) : null;
}

function sync(node) {
    if (!node.__gptImageInputDefs) {
        node.__gptImageInputDefs = Object.fromEntries((node.inputs || []).map(input => [indexOf(input), {...input}]).filter(([index]) => index));
    }
    const highest = Math.max(0, ...(node.inputs || []).filter(input => input.link != null).map(indexOf).filter(Boolean));
    const visible = Math.min(MAX_REFS, Math.max(1, highest + 1));

    for (let index = 1; index <= visible; index++) {
        if (!(node.inputs || []).some(input => indexOf(input) === index)) {
            const input = node.__gptImageInputDefs[index];
            node.addInput(input.name, input.type, input.extra_info);
        }
    }
    for (let slot = (node.inputs || []).length - 1; slot >= 0; slot--) {
        const index = indexOf(node.inputs[slot]);
        if (index > visible && node.inputs[slot].link == null) node.removeInput(slot);
    }
    resizeNode(node);
}

function syncSettings(node) {
    const provider = widget(node, "provider")?.value;
    const model = widget(node, "model");
    if (!MODELS[provider] || !model) return;

    node.__apiImageValues ||= {};
    if (node.__apiImageLastProvider && node.__apiImageLastProvider !== provider) {
        node.__apiImageValues[node.__apiImageLastProvider] = Object.fromEntries(
            PROVIDER_VALUES.map(name => [name, widget(node, name)?.value])
        );
        const saved = node.__apiImageValues[provider];
        for (const name of PROVIDER_VALUES) {
            const item = widget(node, name);
            if (item) item.value = saved?.[name] ?? (name === "api_key" ? "" : item.value);
        }
    }
    node.__apiImageLastProvider = provider;

    setChoices(model, MODELS[provider]);
    const gemini = provider === "Nano Banana (Gemini)";
    const flash31 = model.value === "gemini-3.1-flash-image";
    const gpt25 = model.value.startsWith("gpt-image-2.5-");

    setChoices(widget(node, "aspect_ratio"), gemini && flash31
        ? BASE_RATIOS.concat(EXTENDED_RATIOS)
        : gemini ? BASE_RATIOS : ["match_input", "auto", ...BASE_RATIOS.slice(1)]);
    setChoices(widget(node, "resolution"), gemini && flash31
        ? ["0.5K", "1K", "2K", "4K"]
        : gemini && model.value === "gemini-2.5-flash-image" ? ["1K"] : ["1K", "2K", "4K"]);
    setChoices(widget(node, "quality"), gpt25
        ? ["auto", "low", "medium", "high", "xhigh", "max"]
        : ["auto", "low", "medium", "high"]);
    setChoices(widget(node, "background"), widget(node, "output_format")?.value === "jpeg"
        ? ["auto", "opaque"] : ["auto", "opaque", "transparent"]);

    GPT_WIDGETS.forEach(name => showWidget(widget(node, name), !gemini));
    GEMINI_WIDGETS.forEach(name => showWidget(widget(node, name), gemini));
    showWidget(widget(node, "output_compression"), !gemini && ["jpeg", "webp"].includes(widget(node, "output_format")?.value));
    showWidget(widget(node, "thinking_mode"), gemini && flash31);
    showWidget(widget(node, "search_grounding"), gemini && model.value !== "gemini-2.5-flash-image");

    node.__apiImageValues[provider] = Object.fromEntries(
        PROVIDER_VALUES.map(name => [name, widget(node, name)?.value])
    );

    resizeNode(node);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function hookSettings(node) {
    if (node.__apiImageSettingsHooked) return;
    node.__apiImageSettingsHooked = true;
    for (const name of ["provider", "model", "output_format"]) {
        const item = widget(node, name);
        if (!item) continue;
        const oldCallback = item.callback;
        item.callback = function () {
            const result = oldCallback?.apply(this, arguments);
            syncSettings(node);
            return result;
        };
    }
    syncSettings(node);
}

app.registerExtension({
    name: "APIImageGen.ProgressiveRefInputs",
    async beforeRegisterNodeDef(nodeType) {
        const className = nodeType.comfyClass || nodeType.ComfyClass;
        if (![TARGET, STACKER].includes(className)) return;
        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = oldCreated?.apply(this, arguments);
            if (className === TARGET) {
                sync(this);
                hookSettings(this);
            } else {
                resizeNode(this);
            }
            return result;
        };
        const oldConnections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = oldConnections?.apply(this, arguments);
            if (className === TARGET) sync(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        if ([TARGET, STACKER].includes(node.comfyClass)) {
            if (node.comfyClass === TARGET) {
                sync(node);
                hookSettings(node);
            } else {
                resizeNode(node);
            }
        }
    },
    async afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) {
            if ([TARGET, STACKER].includes(node.comfyClass)) {
                if (node.comfyClass === TARGET) {
                    sync(node);
                    hookSettings(node);
                    syncSettings(node);
                } else {
                    resizeNode(node);
                }
            }
        }
    },
});

