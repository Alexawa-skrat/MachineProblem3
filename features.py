"""Shared feature engineering so training and inference are identical (MP2 pipeline)."""
import numpy as np
import pandas as pd

TARGET = "rating"
FEATURES = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce MP2: original cols, then per-col log1p, and pairwise _plus_/_times_ in loop order.

    8 originals + 8 logs + 28 pairs x 2 = 72 features.
    """
    X = df[FEATURES].copy()
    numeric_cols = list(FEATURES)
    new_cols = {}
    for i, col1 in enumerate(numeric_cols):
        new_cols[f"{col1}_log"] = np.log1p(X[col1])
        for col2 in numeric_cols[i + 1:]:
            new_cols[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
            new_cols[f"{col1}_times_{col2}"] = X[col1] * X[col2]
    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    # Same as MP2 (no categorical predictors exist, so this is a no-op kept for fidelity)
    X = pd.get_dummies(X, drop_first=True)
    return X
