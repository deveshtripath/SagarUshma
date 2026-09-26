"""
Static/mock "last 14 days" ocean reference data.

TODO (later): replace `get_last_14_days` with a real fetch of historical
SST / SSS / SSH / current / wind values for the given (latitude, longitude)
over the 14 days ending on `date` - e.g. from a stored dataset, a NetCDF
archive, or an external ocean-data source. For now the numbers are produced
with a simple deterministic formula (same lat/lon/date always gives the
same series) so the rest of the API has something stable to run against.
"""
import math

# The 7 physical variables we need "last 14 days of" for a given point,
# in addition to lat/lon/month, before we can call the model.
REFERENCE_FIELDS = [
    "sst", "sss", "ssh",
    "u_current", "v_current",
    "u_wind", "v_wind",
]

# (base_value, amplitude, phase_offset) - loosely realistic ranges only,
# not physically meaningful yet.
_PROFILES = {
    "sst":       (20.0, 3.0, 0.0),
    "sss":       (35.0, 0.3, 1.0),
    "ssh":       (0.0, 0.05, 2.0),
    "u_current": (0.1, 0.2, 3.0),
    "v_current": (0.05, 0.2, 4.0),
    "u_wind":    (2.0, 1.5, 5.0),
    "v_wind":    (1.0, 1.5, 6.0),
}


def _value_for_day(latitude, longitude, day_index, base, amplitude, phase_offset):
    """Deterministic, smooth pseudo-value for one field on one day."""
    phase = (latitude * 0.7) + (longitude * 0.3) + phase_offset
    angle = 2 * math.pi * (day_index + phase) / 14.0
    # sst also loosely warmer near the equator, just for plausibility
    lat_adjust = (30 - abs(latitude)) * 0.15 if base == 20.0 else 0.0
    return round(base + lat_adjust + amplitude * math.sin(angle), 4)


def get_last_14_days(latitude, longitude, date):
    """
    Returns {field_name: [14 floats]} for the 14 days ending on `date`
    (index 0 = 13 days before `date`, index 13 = `date` itself).

    Static/placeholder implementation - see module docstring.
    """
    series = {}
    for field, (base, amplitude, phase_offset) in _PROFILES.items():
        series[field] = [
            _value_for_day(latitude, longitude, day_index, base, amplitude, phase_offset)
            for day_index in range(14)
        ]
    return series
