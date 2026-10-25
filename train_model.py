"""Steps 1-3: Read the training dataset, train the model, save it with Pickle."""
import csv
import pickle

from delivery_model_def import DeliveryTimeModel

DATASET_FILE = "delivery_dataset.csv"
MODEL_FILE = "delivery_model.pkl"

records = []
with open(DATASET_FILE, newline="") as f:
    for row in csv.DictReader(f):
        records.append(
            (
                float(row["distance_km"]),
                float(row["weight_kg"]),
                row["transportation"].strip(),
                float(row["delivery_minutes"]),
            )
        )
print(f"Loaded {len(records)} delivery records from {DATASET_FILE}")

model = DeliveryTimeModel().fit(records)

# Report how well the model fits the training data
errors = [abs(model.predict(d, w, t) - m) for d, w, t, m in records]
print(f"Average error on training data: {sum(errors) / len(errors):.2f} minutes")

for transport, (b0, b1, b2) in model.coefficients.items():
    print(f"  {transport}: {b0:.1f} + {b1:.2f}*km + {b2:.2f}*kg")

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_FILE}")
