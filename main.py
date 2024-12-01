from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def save_best_cv_results(best_params: dict, best_score: int, output_file="output/best_cv_results.csv"):
    best_params["best_score"] = best_score

    best_results = pd.DataFrame([best_params])

    best_results.to_csv(output_file, index=False)
    print(f"Best parameters and score saved to {output_file}")


def visualize_all_parameters(estimator: Any, param_grid: dict, X_train: DataFrame, y_train: DataFrame, best_params: dict):
    for param_name, param_values in param_grid.items():
        scores = []

        for value in param_values:
            params = best_params.copy()
            params[param_name] = value

            estimator.set_params(**params)
            estimator.fit(X_train, y_train)

            score = estimator.score(X_train, y_train)
            scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, scores, marker="o", linestyle="--", color="b")
        plt.title(f"Effect of {param_name} on Model Performance", fontsize=14)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel("R-squared Score", fontsize=12)
        plt.grid(True)
        plt.savefig(f'output/images/{param_name}.png')
        plt.show()


def visualisation(y_v: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_v)), y_v, color='g', alpha=0.5)
    plt.title('Scatter Plot of y')
    plt.xlabel('Index')
    plt.ylabel('y values')
    plt.grid(True)
    plt.savefig(f'output/images/public_y')
    plt.show()


def prepare_transformer(num_c: list[int], cat_c: list[int], train_data: DataFrame) -> ColumnTransformer:
    pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_c),
            ('num', pipe, num_c)
        ])

    return preprocessor.fit(train_data)


def get_best_estimator(estimator: Any, param_grid: (dict | list)) -> Any:
    cs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='r2')
    cs.fit(X_train_transformed, y_train.values.ravel())

    save_best_cv_results(best_params=cs.best_params_, best_score=cs.best_score_)

    return cs.best_estimator_


if __name__ == '__main__':
    # ========== Data Loading ==========
    # load input
    X = np.load("input/X_public.npy", allow_pickle=True)
    y = np.load("input/y_public.npy", allow_pickle=True)

    # transform input to DataFrame
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # visualize given y
    visualisation(y)

    # divide input into test and validate parts
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=2)

    # ========== Data Preprocessing ==========
    # divide features into numerical and categorical parts
    numerical_columns = list(range(10, X.shape[1]))
    categorical_columns = list(range(0, 10))

    # transform public input
    transformer = prepare_transformer(num_c=numerical_columns, cat_c=categorical_columns, train_data=X_train)
    X_train_transformed = transformer.transform(X_train)
    X_validate_transformed = transformer.transform(X_validate)

    # ========== Model Definition and Training ==========
    # define model
    model = Ridge()

    # cross validation
    params_grid = {
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
        'fit_intercept': [True, False],
        'solver': ['auto', 'lsqr', 'cholesky'],
        'max_iter': [500, 1000, 5000],
    }
    best_model = get_best_estimator(estimator=model, param_grid=params_grid)

    # Predict y for validation
    y_validate_pred = best_model.predict(X_validate_transformed)

    # Get accuracy score in r2 and mse metrics
    mse = mean_squared_error(y_validate, y_validate_pred)
    r2 = r2_score(y_validate, y_validate_pred)
    # print results
    print(f"Validation Mean Squared Error: {mse}")
    print(f"Validation R-squared: {r2}")

    # visualize the impact of parameters for Ridge
    visualize_all_parameters(best_model, params_grid, X_train_transformed, y_train, best_model.get_params())

    # ========== Working with eval input ==========
    # Read input
    X_eval = np.load("input/X_eval.npy", allow_pickle=True)
    X_eval = pd.DataFrame(X_eval)

    # preprocess input
    X_eval_transformed = transformer.transform(X_eval)

    # predict y_eval
    y_eval = best_model.predict(X_eval_transformed)

    # save y_eval
    np.save('output/y_predikcia.npy', y_eval)
