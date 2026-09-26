# import os, torch
# from huggingface_hub import hf_hub_download

# path = hf_hub_download("devesh123/SagarUshma", "last_model.pt", token=os.environ.get("HF_TOKEN"))
# ckpt = torch.load(path, map_location="cpu", weights_only=True)

# print(type(ckpt))
# if isinstance(ckpt, dict):
#     print([k for k in ckpt.keys() if not k.startswith(("modalities", "fusion", "temporal", "decoder"))])
#     sd = next((ckpt[k] for k in ("model_state_dict", "state_dict") if k in ckpt), ckpt)
# else:
#     sd = ckpt

# for k, v in sd.items():
#     if hasattr(v, "shape"):
#         print(k, tuple(v.shape))


import os, torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("devesh123/SagarUshma", "last_model.pt", token=os.environ.get("HF_TOKEN"))
ckpt = torch.load(path, map_location="cpu", weights_only=True)
sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

for k, v in sd.items():
    print(k, tuple(v.shape))