import json
import logging
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

from .model_def import OUTPUTS
from .model_loader import run
from .reference_data import get_last_14_days

log = logging.getLogger(__name__)


def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


@require_GET
def health(request):
    return JsonResponse({"status": "ok"})


@require_POST
def predict(request):
    # ---- 1. Parse & validate the minimal request body ----
    try:
        data = json.loads(request.body)
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        req_date = _parse_date(data["date"])
    except json.JSONDecodeError:
        return JsonResponse({"error": "Body must be valid JSON"}, status=400)
    except KeyError as e:
        return JsonResponse({"error": f"Missing field: {e.args[0]}"}, status=400)
    except (TypeError, ValueError):
        return JsonResponse(
            {"error": "date must be 'YYYY-MM-DD' and latitude/longitude must be numbers"},
            status=400,
        )

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return JsonResponse(
            {"error": "latitude must be in [-90, 90] and longitude in [-180, 180]"},
            status=400,
        )

    # ---- 2. Last 14 days of sst/sss/ssh/current/wind for this point ----
    # Static/mock for now (see reference_data.py) - will be swapped for a
    # real historical-data fetch later without changing this view.
    history = get_last_14_days(latitude, longitude, req_date)

    # ---- 3. Run the real OceanModel at each standard depth level ----
    try:
        by_depth = run(history)  # list of {"depth":.., "temperature":.., "salinity":..}
    except Exception:
        log.exception("Prediction failed")
        return JsonResponse({"error": "Model error, check server logs"}, status=500)

    # ---- 4. Combine today's (surface) reference values with per-depth output ----
    today_values = {field: series[-1] for field, series in history.items()}
    prediction = {**today_values, "by_depth": by_depth}
    
    return JsonResponse({
        "latitude": latitude,
        "longitude": longitude,
        "date": req_date.isoformat(),
        "prediction": prediction,
        "reference_last_14_days": {
            "sst": history["sst"],
            "sss": history["sss"],
        },
    })