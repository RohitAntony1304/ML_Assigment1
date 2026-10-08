
import argparse

import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander3d
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

ROLL = "IMT2024071"
COLS = ["x1", "x2", "x3"]


def squash(arr, lower, upper):
    return 2.0 * (arr - lower) / (upper - lower) - 1.0


def expand(arr, degree):
    """T_i(x1) T_j(x2) T_k(x3) for every i + j + k <= degree."""
    full = chebvander3d(arr[:, 0], arr[:, 1], arr[:, 2], [degree] * 3)
    i, j, k = np.meshgrid(*[np.arange(degree + 1)] * 3, indexing="ij")
    return full[:, (i + j + k <= degree).ravel()]


def search(scaled, y, degrees, alphas, folds, seed):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for deg in degrees:
        feats = expand(scaled, deg)
        for a in alphas:
            cv = cross_validate(
                Ridge(alpha=a), feats, y, cv=splitter,
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            )
            rows.append({
                "degree": deg,
                "alpha": a,
                "features": feats.shape[1],
                "cv_mse": -cv["test_mse"].mean(),
                "cv_r2": cv["test_r2"].mean(),
            })
            print(f"degree {deg:2d}  alpha {a:<8g}  CV MSE {rows[-1]['cv_mse']:.5f}")
    return pd.DataFrame(rows)


def choose(table, tolerance):
    """Best CV MSE; with tolerance > 0, the lowest degree within that margin."""
    cutoff = table["cv_mse"].min() * (1 + tolerance)
    pool = table[table["cv_mse"] <= cutoff]
    pool = pool[pool["degree"] == pool["degree"].min()]
    return pool.nsmallest(1, "cv_mse").iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"{ROLL}_train_var2.csv")
    ap.add_argument("--test", default=f"{ROLL}_test_var2.csv")
    ap.add_argument("--out", default=f"pred_{ROLL}_var2.csv")
    ap.add_argument("--max-degree", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="0 = pick the lowest CV MSE; 0.02 = simplest degree within 2%%")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    lower = train[COLS].min().to_numpy()
    upper = train[COLS].max().to_numpy()
    scaled_train = squash(train[COLS].to_numpy(), lower, upper)
    scaled_test = squash(test[COLS].to_numpy(), lower, upper)
    y = train["y"].to_numpy()

    alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    table = search(scaled_train, y, range(1, args.max_degree + 1),
                   alpha_grid, args.folds, seed=0)
    table.to_csv(f"cv_grid_{ROLL}_var2.csv", index=False)

    top = choose(table, args.tolerance)
    deg, alpha = int(top["degree"]), float(top["alpha"])

    print("\n===== SELECTED MODEL =====")
    print("Degree  :", deg)
    print("Alpha   :", alpha)
    print("Features:", int(top["features"]))
    print("MSE     :", top["cv_mse"])
    print("R2      :", top["cv_r2"])

    F_train = expand(scaled_train, deg)
    reg = Ridge(alpha=alpha).fit(F_train, y)

    guess = reg.predict(expand(scaled_test, deg))
    result = pd.DataFrame({"y_pred": guess})
    if "y" in test.columns:
        result["y"] = test["y"].to_numpy()

    result.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()