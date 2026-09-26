"""
Static/placeholder depth profile for temperature and salinity.

Why this exists: last_model.pt's depth-conditioned decoder needs
normalization constants (mean/std for inputs, outputs, and depth) that
are NOT present in the checkpoint and haven't been found yet (see the
USE_MODEL_FOR_DEPTH flag in model_loader.py). Until those are found, the
model's raw output is not in physical units (it produced -29 degC and
negative salinity at 1000m - impossible for seawater).

This generates a simple, physically-plausible temperature/salinity-vs-depth
curve anchored to the real surface (sst/sss) reference values, so the API
returns usable numbers for frontend work in the meantime.
"""
import math


def static_depth_profile(surface_temp, surface_salinity, depths):
    """
    Simple thermocline/halocline approximation - NOT real physics, just a
    smooth, bounded curve:
    - temperature relaxes from surface_temp toward ~2 degC (typical deep
      ocean) as depth increases
    - salinity relaxes from surface_salinity toward ~34.7 psu (typical
      deep-ocean value)
    """
    deep_temp = 2.0
    deep_salinity = 34.7
    temp_scale_depth = 400.0    # meters over which most of the drop happens
    sal_scale_depth = 600.0

    profile = []
    for d in depths:
        temp = deep_temp + (surface_temp - deep_temp) * math.exp(-d / temp_scale_depth)
        sal = deep_salinity + (surface_salinity - deep_salinity) * math.exp(-d / sal_scale_depth)
        profile.append({"depth": d, "temperature": round(temp, 4), "salinity": round(sal, 4)})
    return profile