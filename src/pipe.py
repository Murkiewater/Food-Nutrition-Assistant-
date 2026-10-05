from tqdm import tqdm
import numpy as np

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    ParameterGrid,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import accuracy_score, root_mean_squared_error, make_scorer


class TrainValidationTest:

    def __init__(self, test_size=0.2, random_state=21):
        self.test_size = test_size
        self.random_state = random_state

    def get_datasets(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_train,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test


class ModelSelectionRegression:
    def __init__(self, grids: GridSearchCV):
        self.grids = grids
        self.trained = False
        self.model_results = {}

    def choose(self, X_train, y_train, X_valid, y_valid):
        # process of training
        for i, grid in enumerate(self.grids):

            param_grid = grid.param_grid
            estimator = grid.estimator
            cv = grid.cv
            name = estimator.__class__.__name__

            best_score = float("inf")
            # best_estimator = None
            best_params = None

            print(f"Estimator: {name}")
            for params in tqdm(ParameterGrid(param_grid)):
                estimator.set_params(**params)
                score = cross_val_score(
                    estimator,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                )
                mean_score = -np.mean(score)

                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params

            row_results = {
                "name": grid,
                "params": best_params,
                "valid_score": best_score,
            }
            print(f"Best params: {best_params}")
            print(f"Best training RMSE: {best_score:.3f}")

            y_pred = (
                estimator.set_params(**params).fit(X_train, y_train).predict(X_valid)
            )
            valid_score = root_mean_squared_error(y_valid, y_pred)
            print(f"Validation set RMSE score for best params: {valid_score:.3}")
            print()

            self.model_results[i] = row_results

        best_model_name = (
            pd.DataFrame(self.model_results)
            .T.sort_values(by="valid_score", ascending=True)
            .iloc[0]["name"]
        )
        print(f"Classifier with best validation set RMSE: {best_model_name}")

        return best_model_name


def output_cross_val_results(cvs_result, model_name=""):
    print(f"""Cross-validation sore of mode {model_name}
mean = {-np.mean(cvs_result):.2f}, std={np.std(cvs_result):.2f}""")
