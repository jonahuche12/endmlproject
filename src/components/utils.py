import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = get_parameter_grid(model_name)  # Define the parameter grid for each model

            # Create GridSearchCV object
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            grid_search.fit(x_train, y_train)

            # Get the best model from GridSearchCV
            best_model = grid_search.best_estimator_

            # Evaluate the best model on the test set
            y_test_pred = best_model.predict(x_test)
            model_test_score = r2_score(y_test, y_test_pred)

            report[model_name] = model_test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def get_parameter_grid(model_name):
    # Define the parameter grid for each model
    if model_name == "Random Forest":
        return {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == "Decision Tree":
        return {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == "Gradient Boosting":
        return {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    # Add similar blocks for other models

    return {}  # Return an empty dictionary for models without hyperparameters