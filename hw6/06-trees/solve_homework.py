"""Reproduce the answers to ML Zoomcamp 2026 homework 6."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


DATA = Path(__file__).with_name("car_fuel_efficiency_2026.csv")
TARGET = "fuel_efficiency_mpg"


def prepare():
    df = pd.read_csv(DATA).fillna(0)
    full_train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(full_train, test_size=0.25, random_state=1)

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train.drop(columns=TARGET).to_dict(orient="records"))
    x_val = vectorizer.transform(val.drop(columns=TARGET).to_dict(orient="records"))
    y_train = train[TARGET].to_numpy()
    y_val = val[TARGET].to_numpy()
    return df, vectorizer, x_train, x_val, y_train, y_val


def forest_rmse(x_train, x_val, y_train, y_val, n_estimators, max_depth=None):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=1,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return float(root_mean_squared_error(y_val, model.predict(x_val))), model


def main():
    df, vectorizer, x_train, x_val, y_train, y_val = prepare()
    features = vectorizer.get_feature_names_out()

    stump = DecisionTreeRegressor(max_depth=1, random_state=1)
    stump.fit(x_train, y_train)
    split_feature = str(features[stump.tree_.feature[0]])

    sizes = [10, 50, 100, 150]
    depths = [10, 15, 20, 25]
    unrestricted = {}
    for n in sizes:
        unrestricted[n], _ = forest_rmse(x_train, x_val, y_train, y_val, n)

    by_depth = {}
    important_model = None
    for depth in depths:
        scores = {}
        for n in sizes:
            scores[n], model = forest_rmse(x_train, x_val, y_train, y_val, n, depth)
            if depth == 20 and n == 10:
                important_model = model
        by_depth[depth] = scores

    importance = dict(zip(features, map(float, important_model.feature_importances_)))

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(features))
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(features))
    watchlist = [(dtrain, "train"), (dval, "val")]
    xgboost_scores = {}
    for eta in [0.3, 0.1]:
        params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
        }
        history = {}
        model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist,
                          evals_result=history, verbose_eval=False)
        xgboost_scores[eta] = {
            "validation_rmse": float(root_mean_squared_error(y_val, model.predict(dval))),
            "validation_rmse_by_round": history["val"]["rmse"],
        }

    means = {depth: float(np.mean(list(scores.values())))
             for depth, scores in by_depth.items()}
    result = {
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": __import__("sklearn").__version__,
                     "xgboost": xgb.__version__},
        "rows": {"all": len(df), "train": len(y_train), "validation": len(y_val)},
        "q1_split_feature": split_feature,
        "q1_original_column": split_feature.split("=")[0],
        "q2_validation_rmse": unrestricted[10],
        "q3_rmse_by_n_estimators": unrestricted,
        "q3_best_n_estimators_3dp": min(unrestricted, key=lambda n: round(unrestricted[n], 3)),
        "q4_rmse_by_depth_and_estimators": by_depth,
        "q4_mean_rmse_by_depth": means,
        "q4_best_max_depth": min(means, key=means.get),
        "q5_feature_importances": importance,
        "q5_best_of_four": max(
            ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"],
            key=importance.get,
        ),
        "q6_xgboost": xgboost_scores,
        "q6_best_eta": min(xgboost_scores, key=lambda eta: xgboost_scores[eta]["validation_rmse"]),
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for question in ["q1_original_column", "q2_validation_rmse", "q3_best_n_estimators_3dp",
                     "q4_best_max_depth", "q5_best_of_four", "q6_best_eta"]:
        print(f"{question}: {result[question]}")


if __name__ == "__main__":
    main()
