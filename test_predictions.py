"""Activity 2 tests: five delivery records (with and without delay) + history check.
 
Start the server first (python server.py), then run:  python test_predictions.py
"""
import json
import urllib.error
import urllib.request
 
BASE = "http://localhost:5000"
 
TEST_RECORDS = [
    {"customer_name": "Juan Dela Cruz", "distance_km": 25, "weight_kg": 3,
     "transportation": "Motorcycle", "additional_delay_minutes": 20},
    {"customer_name": "Maria Santos", "distance_km": 5, "weight_kg": 2,
     "transportation": "Bicycle", "additional_delay_minutes": 10},
    {"customer_name": "Jose Rizal", "distance_km": 30, "weight_kg": 10,
     "transportation": "Car", "additional_delay_minutes": 45},
    {"customer_name": "Ana Reyes", "distance_km": 12, "weight_kg": 4,
     "transportation": "Motorcycle", "additional_delay_minutes": 0},
    {"customer_name": "Pedro Penduko", "distance_km": 18, "weight_kg": 6,
     "transportation": "Car", "additional_delay_minutes": 15},
]
 
 
def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        return {"HTTP error": e.code, "detail": json.load(e)}
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach the API ({e}). Is server.py running?")
 
 
for i, record in enumerate(TEST_RECORDS, start=1):
    print(f"--- Test {i} ---")
    print(json.dumps(call("/predict", record), indent=2))
 
print("--- Invalid input test (negative delay) ---")
bad = dict(TEST_RECORDS[0], additional_delay_minutes=-5)
print(json.dumps(call("/predict", bad), indent=2))
 
print("--- History saved in delivery_history.csv ---")
for row in call("/history"):
    print(
        f"{row['customer_name']:<15} {row['transportation']:<11} "
        f"base={row['model_predicted_minutes']:>3}  delay={row['additional_delay_minutes']:>3}  "
        f"adjusted={row['adjusted_estimated_minutes']:>3}  {row['delivery_status']}"
    )
 