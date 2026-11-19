
from pathlib import Path
import os
import time
import threading
import json

import serial
import pandas as pd
import joblib
import requests

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

app = Flask(__name__)

# Load WeatherAPI key from .env file
load_dotenv(BASE / ".env", override=True)

API_KEY = os.getenv("WEATHER_API_KEY", "").strip()

print("Weather API key loaded:", bool(API_KEY))


# Single-farm location settings (GPS not required)
FARM_LOCATION_FILE = BASE / "farm_location.json"
DEFAULT_FARM_LOCATION = {"district": "", "state": ""}
location_lock = threading.Lock()
WEATHER_CACHE_DURATION = 120
weather_cache = {}
location_search_cache = {}
location_search_lock = threading.Lock()
weather_cache_lock = threading.Lock()


def load_farm_location():
    try:
        with FARM_LOCATION_FILE.open("r", encoding="utf-8") as file:
            location = json.load(file)
        if isinstance(location, dict) and all(
            isinstance(location.get(k), str) for k in DEFAULT_FARM_LOCATION
        ):
            return {k: location[k].strip() for k in DEFAULT_FARM_LOCATION}
    except (OSError, ValueError, TypeError):
        pass
    return DEFAULT_FARM_LOCATION.copy()


farm_location = load_farm_location()


def location_query(location):
    return ", ".join(
        part for part in (
            location["district"], location["state"], "India"
        ) if part
    )


# ==========================================
# 2. LOAD TRAINED ML MODELS
# ==========================================

crop_model = joblib.load(
    BASE / "models/saved/crop_model.pkl"
)

irrigation_model = joblib.load(
    BASE / "models/saved/irrigation_model.pkl"
)

crop_encoder = joblib.load(
    BASE / "models/saved/crop_encoder.pkl"
)

irrigation_encoder = joblib.load(
    BASE / "models/saved/irrigation_encoder.pkl"
)

yield_model = joblib.load(
    BASE / "models/saved/yield_random_forest_experimental.pkl"
)

print("ML models loaded successfully!")


# ==========================================
# 3. LOAD YIELD DATASET
# ==========================================

yield_dataset = pd.read_csv(
    BASE / "India Agriculture Crop Production.csv",
    usecols=[
        "State",
        "District",
        "Crop",
        "Season",
        "Year",
        "Production Units",
        "Area Units",
        "Area",
        "Production"
    ]
)

# Use records consistent with yield model training
yield_dataset = yield_dataset.dropna(
    subset=[
        "State",
        "District",
        "Crop",
        "Season",
        "Year",
        "Production Units",
        "Area Units",
        "Area",
        "Production"
    ]
).copy()

yield_dataset = yield_dataset[
    (
        yield_dataset["Production Units"]
        .astype(str).str.strip().str.lower() == "tonnes"
    )
    &
    (
        yield_dataset["Area Units"]
        .astype(str).str.strip().str.lower() == "hectare"
    )
    &
    (yield_dataset["Area"] > 0)
    &
    (yield_dataset["Production"] >= 0)
].copy()

for column in ["State", "District", "Crop", "Season"]:
    yield_dataset[column] = (
        yield_dataset[column].astype(str).str.strip()
    )

yield_dataset["Year"] = pd.to_numeric(
    yield_dataset["Year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

yield_dataset = yield_dataset.dropna(subset=["Year"])
yield_dataset["Year"] = yield_dataset["Year"].astype(int)

print("Yield dataset loaded successfully!")


# ==========================================
# 4. ARDUINO CONNECTION
# ==========================================

try:
    arduino = serial.Serial("COM10", 9600, timeout=1)

    time.sleep(2)

    print("Arduino connected successfully!")

except serial.SerialException as e:
    print("Arduino unavailable:", e)
    arduino = None


sensor_data = {
    "temp": None,
    "hum": None,
    "soil": None,
    "crop": "N/A",
    "advice": "N/A"
}


# ==========================================
# 5. FETCH WEATHER DATA
# ==========================================



def find_weather_location(district, state):
    if not API_KEY:
        return None

    key = (district.strip().casefold(), state.strip().casefold())

    with location_search_lock:
        cached = location_search_cache.get(key)

        if cached and time.monotonic() - cached["timestamp"] < 120:
            return cached["coordinates"]

        try:
            response = requests.get(
                "https://api.weatherapi.com/v1/search.json",
                params={
                    "key": API_KEY,
                    "q": f"{district}, {state}, India"
                },
                timeout=10
            )
            response.raise_for_status()

            coordinates = None

            for place in response.json():
                if (
                    place.get("name", "").strip().casefold() == key[0]
                    and place.get("region", "").strip().casefold() == key[1]
                    and place.get("country", "").strip().casefold() == "india"
                ):
                    coordinates = f"{place['lat']},{place['lon']}"
                    break

            location_search_cache[key] = {
                "coordinates": coordinates,
                "timestamp": time.monotonic()
            }

            return coordinates

        except (requests.RequestException, ValueError, KeyError) as error:
            print("Location search error:", error)
            return None




def get_weather(query):
    query = query.strip()
    if not query or not API_KEY:
        print("WEATHER ERROR: location or API key missing")
        return None

    key = query.casefold()
    # Lock covers the request, preventing duplicate calls from concurrent polls.
    with weather_cache_lock:
        cached = weather_cache.get(key)
        if cached and time.monotonic() - cached["timestamp"] < WEATHER_CACHE_DURATION:
            return cached["weather"]

        try:
            response = requests.get(
                "https://api.weatherapi.com/v1/current.json",
                params={"key": API_KEY, "q": query},
                timeout=10
            )
            if not response.ok:
                try:
                    message = response.json().get("error", {}).get("message", "Unknown error")
                except ValueError:
                    message = "Invalid API response"
                print("WEATHER API ERROR:", response.status_code, message)
                return None

            weather = response.json()
            if "current" not in weather:
                print("WEATHER ERROR: Current data missing")
                return None
            weather_cache[key] = {
                "weather": weather,
                "timestamp": time.monotonic()
            }
            print("Weather fetched from API:", weather.get("location", {}).get("name", query))
            return weather
        except (requests.RequestException, ValueError) as error:
            print("WEATHER CONNECTION ERROR:", error)
            return None


# ==========================================
# 6. WEATHER ADVICE
# ==========================================

def generate_weather_advice(weather):

    if weather is None:
        return ["⚠️ Weather data not available"]

    current = weather["current"]

    temp = current["temp_c"]
    humidity = current["humidity"]
    condition = current["condition"]["text"].lower()
    rain = current["precip_mm"]

    advice = []

    if "rain" in condition or rain > 0:
        advice.append(
            "🌧️ Rain detected → Check soil moisture before irrigation"
        )

    if temp > 30:
        advice.append(
            "☀️ High temperature → Monitor crop water requirements"
        )

    if humidity < 60 and rain == 0:
        advice.append(
            "🌾 Low humidity → Monitor soil moisture"
        )

    if humidity > 85:
        advice.append(
            "⚠️ High humidity → Monitor for crop diseases"
        )

    if not advice:
        advice.append(
            "✅ Current weather conditions appear normal"
        )

    return advice


# ==========================================
# 7. ARDUINO SENSOR READING
# ==========================================

def read_arduino():

    while arduino:

        try:
            line = arduino.readline().decode(
                errors="ignore"
            ).strip()

            if not line:
                continue

            t, h, s = map(
                float,
                line.split(",")
            )

            sensor_data["temp"] = t
            sensor_data["hum"] = h
            sensor_data["soil"] = s

            df = pd.DataFrame(
                [[t, h, s]],
                columns=[
                    "Temperature",
                    "Humidity",
                    "SoilMoisture"
                ]
            )

            crop_prediction = crop_model.predict(df)
            irrigation_prediction = irrigation_model.predict(df)

            sensor_data["crop"] = (
                crop_encoder.inverse_transform(
                    crop_prediction
                )[0]
            )

            sensor_data["advice"] = (
                irrigation_encoder.inverse_transform(
                    irrigation_prediction
                )[0]
            )

        except Exception as e:
            print("Sensor reading error:", e)
            time.sleep(1)


if arduino is not None:

    threading.Thread(
        target=read_arduino,
        daemon=True
    ).start()


# ==========================================
# 8. HOME PAGE
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# 9. DASHBOARD DATA API
# ==========================================

@app.route("/data")
def data():
    with location_lock:
        location = farm_location.copy()
    configured = all(location.values())
    
    weather = None

    if configured:
        coordinates = find_weather_location(
            location["district"],
            location["state"]
        )

        if coordinates is not None:
            weather = get_weather(coordinates)

    return jsonify({
        **sensor_data,
        "weather_alerts": (
            generate_weather_advice(weather) if configured
            else ["📍 Save your farm location to see local weather alerts"]
        ),
        "farm_location": location,
        "weather_location": weather.get("location", {}).get("name") if weather else None
    })


@app.route("/farm-location", methods=["GET"])
def read_farm_location():
    with location_lock:
        return jsonify(farm_location.copy())


@app.route("/farm-location", methods=["POST"])
def update_farm_location():
    global farm_location
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Please provide a valid location."}), 400

    location = {}
    for field in ("district", "state"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
            return jsonify({"error": f"Enter a valid {field} (up to 100 characters)."}), 400
        location[field] = value.strip()

    # Verify location before saving, rather than silently showing weather elsewhere.
    
    coordinates = find_weather_location(
        location["district"],
        location["state"]
    )

    if coordinates is None:
        return jsonify({
            "error": "District aur State ka exact match nahi mila. Please check spelling."
        }), 400

    weather = get_weather(coordinates)

    if weather is None:
        return jsonify({
            "error": "WeatherAPI could not verify this location. Check spelling or API access."
        }), 400

    if weather is None:
        return jsonify({"error": "WeatherAPI could not verify this location. Check spelling or API access."}), 400

    temporary_file = FARM_LOCATION_FILE.with_suffix(".json.tmp")
    with location_lock:
        try:
            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(location, file, indent=2)
            os.replace(temporary_file, FARM_LOCATION_FILE)
            farm_location = location.copy()
        except OSError:
            return jsonify({"error": "Could not save location to disk."}), 500
        finally:
            try:
                temporary_file.unlink(missing_ok=True)
            except OSError:
                pass

    return jsonify({
        "message": "Farm location saved.",
        "farm_location": location,
        "weather_location": weather.get("location", {}).get("name")
    })


# ==========================================
# 10. YIELD DROPDOWN OPTIONS
# ==========================================

@app.route("/yield-options")
def yield_options():

    states = sorted(
        yield_dataset["State"].unique().tolist()
    )

    crops = sorted(
        yield_dataset["Crop"].unique().tolist()
    )

    seasons = sorted(
        yield_dataset["Season"].unique().tolist()
    )

    state = request.args.get(
        "state", ""
    ).strip()

    if state:

        districts = sorted(
            yield_dataset.loc[
                yield_dataset["State"] == state,
                "District"
            ].unique().tolist()
        )

    else:
        districts = []

    return jsonify({
        "states": states,
        "districts": districts,
        "crops": crops,
        "seasons": seasons
    })



# Historical availability for Arduino-recommended crop.
@app.route("/yield-availability")
def yield_availability():
    state = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    crop = request.args.get("crop", "").strip()
    if not all((state, district, crop)):
        return jsonify({"crop": None, "seasons": []})

    # Match case/spacing, while retaining the dataset's exact crop label.
    def normalize(series):
        return series.astype(str).str.strip().str.casefold()

    matching = yield_dataset.loc[
        (normalize(yield_dataset["State"]) == state.casefold()) &
        (normalize(yield_dataset["District"]) == district.casefold()) &
        (normalize(yield_dataset["Crop"]) == crop.casefold())
    ]
    if matching.empty:
        return jsonify({"crop": None, "seasons": []})
    return jsonify({
        "crop": str(matching.iloc[0]["Crop"]),
        "seasons": sorted(matching["Season"].unique().tolist())
    })


# ==========================================
# 11. CROP YIELD PREDICTION API
# ==========================================

@app.route("/predict-yield", methods=["POST"])
def predict_yield():

    try:
        data = request.get_json(silent=True) or {}

        state = str(
            data.get("state", "")
        ).strip()

        district = str(
            data.get("district", "")
        ).strip()

        crop = str(
            data.get("crop", "")
        ).strip()

        season = str(
            data.get("season", "")
        ).strip()

        year = int(
            data.get("year", 2026)
        )

        if not all([
            state,
            district,
            crop,
            season
        ]):

            return jsonify({
                "error": "Please select all required fields."
            }), 400

        if year < 1990 or year > 2100:

            return jsonify({
                "error": "Please enter a valid year."
            }), 400

        matching_records = yield_dataset[
            (yield_dataset["State"] == state)
            &
            (yield_dataset["District"] == district)
            &
            (yield_dataset["Crop"] == crop)
            &
            (yield_dataset["Season"] == season)
        ]

        if matching_records.empty:

            return jsonify({
                "error": (
                    "Selected state, district, crop and "
                    "season combination is not available "
                    "in the historical dataset."
                )
            }), 400

        input_df = pd.DataFrame([{
            "State": state,
            "District": district,
            "Crop": crop,
            "Season": season,
            "Year": year
        }])

        predicted_yield = float(
            yield_model.predict(input_df)[0]
        )

        predicted_yield = max(
            0.0,
            predicted_yield
        )

        return jsonify({
            "predicted_yield": round(
                predicted_yield, 2
            ),
            "unit": "tonnes/hectare",
            "crop": crop,
            "district": district,
            "state": state,
            "year": year,
            "note": (
                "Experimental estimate based on historical "
                "district-level data, not a guaranteed farm yield."
            )
        })

    except (ValueError, TypeError):

        return jsonify({
            "error": "Invalid prediction input."
        }), 400

    except Exception as e:

        print("Yield prediction error:", e)

        return jsonify({
            "error": "Unable to generate yield prediction."
        }), 500


# ==========================================
# 12. RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
