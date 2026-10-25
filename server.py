"""Activity 2: Delivery prediction + transport comparison + hypothetical delay + CSV history.
 
Reuses the SAME saved model (delivery_model.pkl). Nothing is retrained and no
base delivery time is hardcoded. Everything runs locally, no internet needed.
 
Run:   python server.py
Open:  http://localhost:5000
"""
import csv
import json
import os
import pickle
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
 
from delivery_model_def import DeliveryTimeModel  # needed so Pickle can load the model
 
MODEL_FILE = "delivery_model.pkl"
HISTORY_FILE = "delivery_history.csv"
HOST, PORT = "localhost", 5000
 
HISTORY_COLUMNS = [
    "timestamp",
    "customer_name",
    "distance_km",
    "weight_kg",
    "transportation",
    "model_predicted_minutes",
    "additional_delay_minutes",
    "adjusted_estimated_minutes",
    "delivery_status",
    "fastest_transportation",
]
 
# Load the existing saved model (NO retraining happens here)
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print(f"Loaded saved model from {MODEL_FILE}")
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGE_FILE = os.path.join(BASE_DIR, "index.html")
 
 
def render_page():
    """Read index.html (the user interface) and insert the vehicle list from the model."""
    try:
        with open(PAGE_FILE, encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return "<h2>index.html is missing. Put it in the same folder as server.py.</h2>"
    return html.replace("__TRANSPORTS__", json.dumps(model.transport_types))
 
 
def validate(data):
    """Return (clean_dict, error_message). Exactly one of them is None."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
 
    name = str(data.get("customer_name", "")).strip()
    if not name:
        return None, "customer_name is required."
 
    try:
        distance = float(data.get("distance_km"))
        weight = float(data.get("weight_kg"))
    except (TypeError, ValueError):
        return None, "distance_km and weight_kg must be numbers."
 
    if not 0 < distance <= 200:
        return None, "distance_km must be greater than 0 and at most 200."
    if not 0 < weight <= 100:
        return None, "weight_kg must be greater than 0 and at most 100."
 
    transport = str(data.get("transportation", "")).strip().title()
    if transport not in model.transport_types:
        return None, f"transportation must be one of: {', '.join(model.transport_types)}."
 
    # The delay is user-supplied. If it is left blank it counts as 0 (no delay).
    raw_delay = data.get("additional_delay_minutes", 0)
    if raw_delay is None or str(raw_delay).strip() == "":
        raw_delay = 0
    try:
        delay = float(raw_delay)
    except (TypeError, ValueError):
        return None, "additional_delay_minutes must be a number."
    if not 0 <= delay <= 1440:
        return None, "additional_delay_minutes must be between 0 and 1440."
 
    return {
        "customer_name": name,
        "distance_km": distance,
        "weight_kg": weight,
        "transportation": transport,
        "additional_delay_minutes": delay,
    }, None
 
 
def tidy(x):
    """Show 25.0 as 25."""
    return int(x) if float(x).is_integer() else x
 
 
def save_history(row):
    """Append one prediction to delivery_history.csv (creates it with a header)."""
    new_file = not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
 
 
def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
 
 
def build_result(clean):
    d, w, t = clean["distance_km"], clean["weight_kg"], clean["transportation"]
 
    # 1. Base prediction from the saved model
    base = round(model.predict(d, w, t))
 
    # 2. Compare ALL transportation types for the same package (also from the model)
    comparison = {tr: round(model.predict(d, w, tr)) for tr in model.transport_types}
    fastest = min(comparison, key=comparison.get)
 
    # 3. Add the hypothetical delay and calculate the adjusted time
    delay = clean["additional_delay_minutes"]
    adjusted = base + delay
    status = "DELAY ADDED" if delay > 0 else "NO DELAY"
 
    result = {
        "customer_name": clean["customer_name"],
        "distance_km": tidy(d),
        "weight_kg": tidy(w),
        "transportation": t,
        "model_predicted_minutes": base,
        "additional_delay_minutes": tidy(delay),
        "adjusted_estimated_minutes": tidy(adjusted),
        "delivery_status": status,
        "transport_comparison": comparison,
        "fastest_transportation": fastest,
    }
 
    # 4. Save to CSV history
    save_history(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_name": result["customer_name"],
            "distance_km": result["distance_km"],
            "weight_kg": result["weight_kg"],
            "transportation": t,
            "model_predicted_minutes": base,
            "additional_delay_minutes": result["additional_delay_minutes"],
            "adjusted_estimated_minutes": result["adjusted_estimated_minutes"],
            "delivery_status": status,
            "fastest_transportation": fastest,
        }
    )
    return result
 
 
class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
 
    def do_GET(self):
        if self.path == "/":
            self._send(200, render_page(), "text/html")
        elif self.path == "/history":
            self._send(200, json.dumps(read_history(), indent=2))
        else:
            self._send(404, json.dumps({"error": "Not found"}))
 
    def do_POST(self):
        if self.path != "/predict":
            self._send(404, json.dumps({"error": "Not found"}))
            return
 
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "Invalid JSON."}))
            return
 
        clean, error = validate(data)
        if error:
            self._send(400, json.dumps({"error": error}, indent=2))
            return
 
        self._send(200, json.dumps(build_result(clean), indent=2))
 
    def log_message(self, fmt, *args):
        print(f"[request] {self.address_string()} - {fmt % args}")
 
 
if __name__ == "__main__":
    print(f"Server running at http://{HOST}:{PORT}  (press Ctrl+C to stop)")
    HTTPServer((HOST, PORT), Handler).serve_forever()
 