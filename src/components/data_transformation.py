import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.components.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_path = DataTransformationConfig()

    def get_data_transformation_obj(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),  # Set with_mean=False for sparse input
                ]
            )


            logging.info("Numerical Column Standard Scaling Completed")
            logging.info("Categorical Column encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data reading Completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformation_obj()
            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_input_feature_df = train_df.drop(columns=[target_column], axis=1)
            train_target_feature = train_df[target_column]

            test_input_feature_df = test_df.drop(columns=[target_column], axis=1)
            test_target_feature = test_df[target_column]

            logging.info("Applying preprocessor object on Train and test Dataframes")

            train_input_arr = preprocessing_obj.fit_transform(train_input_feature_df)
            test_input_arr = preprocessing_obj.transform(test_input_feature_df)

            train_arr = np.c_[train_input_arr, np.array(train_target_feature).reshape(-1, 1)]
            test_arr = np.c_[test_input_arr, np.array(test_target_feature).reshape(-1, 1)]
            logging.info("Saved Preprocessor Object")

            save_object(
                file_path=self.data_tranformation_path.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tranformation_path.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)
