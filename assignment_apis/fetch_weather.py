import json
from datetime import date, timedelta
from pathlib import Path

import requests
import config

def geocode_city(city: str) -> tuple[float, float]:
    """
    Fetch coordinates using Open-Meteo Geocoding API.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    if not data.get("results"):
        raise RuntimeError(f"City not found: {city}")
    
    result = data["results"][0]
    return float(result["latitude"]), float(result["longitude"])

def fetch_weather_history(lat: float, lon: float, days_back: int, units: str) -> list:
    """
    Fetch historical weather data for a range of days in a single request.
    """
    # Open-Meteo can take start/end dates
    end_day = date.today() - timedelta(days=1)
    start_day = end_day - timedelta(days=days_back - 1)
    
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Map units if necessary (Open-Meteo uses 'celsius' or 'fahrenheit')
    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": temp_unit,
        "timezone": "auto"
    }
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    # Process the parallel arrays into a list of daily dictionaries
    daily_data = data.get("daily", {})
    results = []
    for i in range(len(daily_data.get("time", []))):
        results.append({
            "date": daily_data["time"][i],
            "max_temp": daily_data["temperature_2m_max"][i],
            "min_temp": daily_data["temperature_2m_min"][i],
            "precipitation": daily_data["precipitation_sum"][i]
        })
    
    return results

def main() -> None:
    # Load settings from config
    city = getattr(config, "CITY_NAME", "Madrid")
    country = getattr(config, "COUNTRY_CODE", "ES")
    days_back = int(getattr(config, "DAYS_BACK", 10))
    units = getattr(config, "UNITS", "metric")

    # Setup paths
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "raw_weather.json"

    # Step 1: Get coordinates
    print(f"Geocoding: {city}...")
    lat, lon = geocode_city(city)

    # Step 2: Fetch all days in one batch
    print(f"Fetching weather for last {days_back} days...")
    daily_results = fetch_weather_history(lat, lon, days_back, units)

    # Step 3: Prepare output
    out = {
        "source": "open_meteo_api",
        "city": {"name": city, "country": country, "lat": lat, "lon": lon},
        "date_range": {
            "start": daily_results[0]["date"], 
            "end": daily_results[-1]["date"], 
            "days": len(daily_results)
        },
        "units": units,
        "daily": daily_results,
    }

    # Step 4: Save to file
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_path} ({len(daily_results)} days)")

if __name__ == "__main__":
    main()