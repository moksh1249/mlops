"""Run Practical 05 end to end and report the task self-check results."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
DATA = ROOT / "data" / "delivery_times.csv"
SEED = 42
N_ROWS = 600


def make_delivery_csv(path: Path = DATA) -> Path:
    """Create the deterministic delivery dataset used by the practical."""
    rng = np.random.default_rng(SEED)
    distance_km = np.round(rng.uniform(0.5, 12.0, N_ROWS), 2)
    prep_time_min = np.round(rng.uniform(5, 30, N_ROWS), 0)
    traffic_level = rng.integers(1, 4, N_ROWS)
    rain = rng.binomial(1, 0.25, N_ROWS)
    delivery_min = np.round(
        6.0
        + 3.1 * distance_km
        + 0.65 * prep_time_min
        + 4.2 * traffic_level
        + 5.5 * rain
        + rng.normal(0, 2.5, N_ROWS),
        1,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            ["distance_km", "prep_time_min", "traffic_level", "rain", "delivery_min"]
        )
        for index in range(N_ROWS):
            writer.writerow(
                [
                    distance_km[index],
                    int(prep_time_min[index]),
                    int(traffic_level[index]),
                    int(rain[index]),
                    delivery_min[index],
                ]
            )
    return path


def write_project_files() -> None:
    """Write the config, module, fixtures, and completed student tests."""
    WORK.mkdir(parents=True, exist_ok=True)

    config = {
        "data": {
            "path": "../data/delivery_times.csv",
            "features": ["distance_km", "prep_time_min", "traffic_level", "rain"],
            "target": "delivery_min",
        },
        "split": {"test_size": 0.2, "seed": SEED},
        "model": {"name": "linear_regression"},
        "training": {"max_rows": 300},
    }
    (WORK / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    (WORK / "orders.py").write_text(
        '''"""Small helpers for delivery orders."""

TRAFFIC_LEVELS = (1, 2, 3)


def minutes_per_km(delivery_min, distance_km):
    """Return the number of delivery minutes per kilometre."""
    if distance_km <= 0:
        raise ValueError("distance_km must be positive")
    return delivery_min / distance_km


def is_valid_order(order):
    """Return True when an order is usable for training."""
    if order.get("distance_km", 0) <= 0:
        return False
    if order.get("prep_time_min", 0) < 0:
        return False
    if order.get("traffic_level") not in TRAFFIC_LEVELS:
        return False
    if order.get("rain") not in (0, 1):
        return False
    return True
''',
        encoding="utf-8",
    )

    (WORK / "test_orders.py").write_text(
        '''"""Tests for orders.py."""

import pytest

from orders import is_valid_order, minutes_per_km


def test_minutes_per_km_simple():
    assert minutes_per_km(30, 10) == 3.0


def test_minutes_per_km_rejects_zero_distance():
    with pytest.raises(ValueError):
        minutes_per_km(30, 0)


def test_good_order_is_accepted():
    good = {"distance_km": 5.0, "prep_time_min": 20,
            "traffic_level": 2, "rain": 0}
    assert is_valid_order(good) is True


def test_bad_traffic_level_is_rejected():
    bad = {"distance_km": 5.0, "prep_time_min": 20,
           "traffic_level": 9, "rain": 0}
    assert is_valid_order(bad) is False
''',
        encoding="utf-8",
    )

    (WORK / "test_speed.py").write_text(
        '''"""Student tests for minutes_per_km."""

import pytest

from orders import minutes_per_km


def test_slow_delivery():
    assert minutes_per_km(60, 6) == 10.0


def test_negative_distance_raises():
    with pytest.raises(ValueError):
        minutes_per_km(60, -3)
''',
        encoding="utf-8",
    )

    (WORK / "test_model.py").write_text(
        '''"""Behaviour tests for the trained model."""

import pandas as pd


def make_order(distance_km=5.0, prep_time_min=20,
               traffic_level=2, rain=0):
    return pd.DataFrame([{
        "distance_km": distance_km,
        "prep_time_min": prep_time_min,
        "traffic_level": traffic_level,
        "rain": rain,
    }])


def test_rain_never_makes_delivery_faster(trained_model):
    dry = trained_model.predict(make_order(rain=0))[0]
    wet = trained_model.predict(make_order(rain=1))[0]
    assert wet >= dry


def test_further_is_never_faster(trained_model):
    near = trained_model.predict(make_order(distance_km=2))[0]
    far = trained_model.predict(make_order(distance_km=12))[0]
    assert far > near


def test_error_is_better_than_guessing(model_mae, baseline_mae):
    assert model_mae < baseline_mae


def test_heavier_traffic_is_never_faster(trained_model):
    light = trained_model.predict(make_order(traffic_level=1))[0]
    heavy = trained_model.predict(make_order(traffic_level=3))[0]
    assert heavy >= light
''',
        encoding="utf-8",
    )

    (WORK / "conftest.py").write_text(
        '''"""Shared pytest fixtures for the practical."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def config():
    with (HERE / "config.yaml").open(encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


@pytest.fixture(scope="session")
def split(config):
    data_path = HERE.parent / "data" / "delivery_times.csv"
    orders = pd.read_csv(data_path)
    X = orders[config["data"]["features"]]
    y = orders[config["data"]["target"]]
    return train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["seed"],
    )


@pytest.fixture(scope="session")
def trained_model(split):
    X_tr, _, y_tr, _ = split
    return LinearRegression().fit(X_tr, y_tr)


@pytest.fixture(scope="session")
def model_mae(trained_model, split):
    _, X_te, _, y_te = split
    return mean_absolute_error(y_te, trained_model.predict(X_te))


@pytest.fixture(scope="session")
def baseline_mae(split):
    _, X_te, y_tr, y_te = split
    guess = [y_tr.mean()] * len(y_te)
    return mean_absolute_error(y_te, guess)
''',
        encoding="utf-8",
    )


def train_limited(config: dict, data_path: Path) -> float:
    """Train on the configured number of rows and return the test MAE."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    orders = pd.read_csv(data_path).head(config["training"]["max_rows"])
    features = config["data"]["features"]
    target = config["data"]["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        orders[features],
        orders[target],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["seed"],
    )
    model = LinearRegression().fit(X_train, y_train)
    return float(mean_absolute_error(y_test, model.predict(X_test)))


def run_tests() -> subprocess.CompletedProcess[str]:
    """Run every generated test with the active virtual environment."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=WORK,
        capture_output=True,
        text=True,
        check=False,
    )


def print_task_outputs(config: dict, limited_mae: float) -> None:
    """Print the observable outputs produced by T1, T2, and T3."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    orders = pd.read_csv(DATA)
    features = config["data"]["features"]
    target = config["data"]["target"]
    X_train, _, y_train, _ = train_test_split(
        orders[features],
        orders[target],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["seed"],
    )
    model = LinearRegression().fit(X_train, y_train)

    def predict(**values: float) -> float:
        return float(model.predict(pd.DataFrame([values], columns=features))[0])

    print("\n" + "=" * 66)
    print("TASK 1 - CONFIGURATION AND LIMITED TRAINING")
    print("=" * 66)
    print(f"Configured training rows: {config['training']['max_rows']}")
    print(f"Features: {', '.join(features)}")
    print(f"Target: {target}")
    print(f"MAE using configured rows: {limited_mae:.2f}")
    print(f"Dataset rows available: {len(orders)}")

    print("\n" + "=" * 66)
    print("TASK 2 - STUDENT TESTS")
    print("=" * 66)
    print("Generated work/test_speed.py:")
    print((WORK / "test_speed.py").read_text(encoding="utf-8").strip())

    print("\n" + "=" * 66)
    print("TASK 3 - MODEL BEHAVIOUR")
    print("=" * 66)
    base_order = {
        "distance_km": 5.0,
        "prep_time_min": 20,
        "traffic_level": 2,
        "rain": 0,
    }
    dry = predict(**base_order)
    wet = predict(**{**base_order, "rain": 1})
    near = predict(**{**base_order, "distance_km": 2})
    far = predict(**{**base_order, "distance_km": 12})
    light = predict(**{**base_order, "traffic_level": 1})
    heavy = predict(**{**base_order, "traffic_level": 3})
    print(f"Dry order prediction: {dry:.2f} minutes")
    print(f"Rainy order prediction: {wet:.2f} minutes")
    print(f"Near order prediction: {near:.2f} minutes")
    print(f"Far order prediction: {far:.2f} minutes")
    print(f"Light-traffic prediction: {light:.2f} minutes")
    print(f"Heavy-traffic prediction: {heavy:.2f} minutes")


def self_check(config: dict, t1_mae: float,
               test_run: subprocess.CompletedProcess[str]) -> bool:
    """Print and return the final practical self-check result."""
    speed_tests = (WORK / "test_speed.py").read_text(encoding="utf-8")
    model_tests = (WORK / "test_model.py").read_text(encoding="utf-8")
    checks = [
        ("T1 config has training.max_rows = 300",
         config["training"]["max_rows"] == 300),
        ("T1 limited training returns a sensible MAE",
         0 < t1_mae < 15),
        ("T1 uses only the configured rows",
         abs(t1_mae - train_limited({**config, "training": {"max_rows": 600}}, DATA)) > 1e-9),
        ("T2 test_speed.py contains both required tests",
         "def test_slow_delivery" in speed_tests
         and "def test_negative_distance_raises" in speed_tests),
        ("T2 tests contain assert and pytest.raises",
         "assert" in speed_tests and "pytest.raises" in speed_tests),
        ("T2 pytest run exits with code 0", test_run.returncode == 0),
        ("T2 and T3 collect at least 9 tests",
         any(f"{count} passed" in test_run.stdout for count in range(9, 40))),
        ("T3 traffic test uses trained_model and assert",
         "test_heavier_traffic_is_never_faster(trained_model)" in model_tests
         and model_tests.rsplit("def test_heavier", 1)[1].count("assert") >= 1),
        ("T3 full suite still passes", test_run.returncode == 0),
    ]

    print("=" * 66)
    print("SELF-CHECK  Practical 05")
    print("=" * 66)
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {label}")
    passed_count = sum(passed for _, passed in checks)
    print("-" * 66)
    print(f"  {passed_count} of {len(checks)} checks passed")
    print("=" * 66)
    if test_run.stdout:
        print(test_run.stdout.strip())
    if test_run.stderr:
        print(test_run.stderr.strip())
    return passed_count == len(checks)


def main() -> int:
    make_delivery_csv()
    write_project_files()

    with (WORK / "config.yaml").open(encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    t1_mae = train_limited(config, DATA)
    print_task_outputs(config, t1_mae)

    test_run = run_tests()
    return 0 if self_check(config, t1_mae, test_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())