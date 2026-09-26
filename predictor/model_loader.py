"""Downloads the model from Hugging Face once, keeps it in memory, and runs predictions."""
import math
import os
import threading

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from .depth_profile import static_depth_profile

from .model_def import OUTPUT_SIZE, OUTPUTS, PATCH_SIZE, OceanModel

_lock = threading.Lock()
_cache = {}

# Flip to True once real normalization stats (mean/std for inputs,
# outputs, and depth) are found and wired in below - until then the
# model's raw output is not in physical units (see chat: -29 degC and
# negative salinity at depth=1000).
USE_MODEL_FOR_DEPTH = False

def _load():
    path = hf_hub_download(
        repo_id=os.environ["HF_REPO_ID"],
        filename=os.environ["HF_MODEL_FILENAME"],
        token=os.environ.get("HF_TOKEN"),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    state = {k.removeprefix("module."): v for k, v in state.items()}

    model = OceanModel()
    model.load_state_dict(state)   # will raise loudly if shapes/keys ever drift again
    model.eval()
    return {"model": model}


def _get():
    if "bundle" not in _cache:
        with _lock:
            if "bundle" not in _cache:
                _cache["bundle"] = _load()  # first request downloads + loads (slow once)
    return _cache["bundle"]


def _patch(values):
    """Broadcast a handful of per-channel scalars into a (1, C, H, W) patch.

    Placeholder: we don't have real gridded data yet, so every pixel in the
    patch gets the same value. Swap this for a real spatial patch once the
    14-day reference data becomes real (see reference_data.py).
    """
    arr = np.zeros((1, len(values), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i, v in enumerate(values):
        arr[0, i, :, :] = v
    return torch.from_numpy(arr)

# Standard depth levels (meters) to report per prediction.
# ASSUMPTION - these do NOT necessarily match the depths your model was
# trained on. Replace with your training notebook's actual depth levels
# (e.g. GLORYS standard depths) for numerically correct results.
DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def run(history, depths=None):
    """
    history: dict from reference_data.get_last_14_days ->
             {"sst":[14], "sss":[14], "ssh":[14], "u_current":[14],
              "v_current":[14], "u_wind":[14], "v_wind":[14]}
    depths: list of depths (meters) to report. Defaults to DEPTH_LEVELS.

    Returns: list of dicts, one per depth:
             [{"depth": 0, "temperature": ..., "salinity": ...}, ...]
    """
    depths = depths if depths is not None else DEPTH_LEVELS
    surface_sst = history["sst"][-1]
    surface_sss = history["sss"][-1]

    if not USE_MODEL_FOR_DEPTH:
        # Static/placeholder profile - see depth_profile.py. Skips the
        # model entirely (no HF download, no torch inference) until real
        # normalization stats are found.
        return static_depth_profile(surface_sst, surface_sss, depths)

    # ---- real model path (re-enable once normalization is fixed) ----
    b = _get()
    model = b["model"]
    sequence = []
    for day in range(14):
        u_c, v_c = history["u_current"][day], history["v_current"][day]
        u_w, v_w = history["u_wind"][day], history["v_wind"][day]
        sequence.append({
            "sst": _patch([history["sst"][day]] * 3),
            "sss": _patch([history["sss"][day]]),
            "ssh": _patch([history["ssh"][day]]),
            "current": _patch([u_c, v_c, math.hypot(u_c, v_c)]),
            "wind": _patch([u_w, v_w, math.hypot(u_w, v_w)]),
        })

    center = OUTPUT_SIZE // 2
    results = []
    with torch.inference_mode():
        hidden = model.encode(sequence)   # computed once, reused per depth
        for d in depths:
            depth_t = torch.tensor([[float(d)]], dtype=torch.float32)
            out = model.decode(hidden, depth_t).numpy()[0]
            values = {name: float(out[c, center, center]) for c, name in enumerate(OUTPUTS)}
            results.append({"depth": d, **values})
    return results