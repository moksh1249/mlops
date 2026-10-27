"Predict one delivery, from the command line."

import pandas as pd
from pathlib import Path

from delivery import load_model

HERE = Path(__file__).resolve().parent

def main():
    # TODO: load work/model.joblib, predict one 7 km / 25 min /
    #       traffic 3 / no rain order, and print "PREDICTION: <minutes>"
    model_path = HERE / "model.joblib"
    model = load_model(model_path)

    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    minutes = model.predict(order)[0]

    print(f"PREDICTION: {minutes:.1f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
