
import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = [f"x{n}" for n in range(1, 7)]
TARGET = "y"


def build_pipeline(degree, alpha, iters):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=iters),
    )


def sweep(X, y, degrees, alphas, folds=3, seed=0):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for deg in degrees:
        for a in alphas:
            cv = cross_validate(
                build_pipeline(deg, a, 3000), X, y, cv=splitter,
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            )
            rows.append({
                "degree": deg,
                "alpha": a,
                "mse": -cv["test_mse"].mean(),
                "r2": cv["test_r2"].mean(),
            })
    return pd.DataFrame(rows)


def pick_simplest(table, tolerance=0.02):
    """Lowest degree within `tolerance` of the best MSE; best alpha at that degree."""
    cutoff = table["mse"].min() * (1 + tolerance)
    shortlist = table[table["mse"] <= cutoff]
    lowest = shortlist["degree"].min()
    return shortlist[shortlist["degree"] == lowest].nsmallest(1, "mse").iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", default="IMT2024071")
    ap.add_argument("--max-degree", type=int, default=6)
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    folder = Path(args.data_dir)
    train = pd.read_csv(folder / f"{args.roll}_train_var1.csv")
    test = pd.read_csv(folder / f"{args.roll}_test_var1.csv")

    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    table = sweep(train[FEATURES], train[TARGET],
                  range(1, args.max_degree + 1), alpha_grid)
    top = pick_simplest(table)

    print(f"Degree: {int(top['degree'])}")
    print(f"Alpha : {top['alpha']}")
    print(f"CV MSE: {top['mse']}")
    print(f"CV R2 : {top['r2']}")

    final = build_pipeline(int(top["degree"]), top["alpha"], 10000)
    final.fit(train[FEATURES], train[TARGET])

    result = pd.DataFrame({"y_pred": final.predict(test[FEATURES])})
    if TARGET in test.columns:
        result[TARGET] = test[TARGET].to_numpy()

    dest = f"pred_{args.roll}_var1.csv"
    result.to_csv(dest, index=False)
    table.to_csv(f"cv_grid_{args.roll}_var1.csv", index=False)
    print("Saved:", dest)


if __name__ == "__main__":
    main()