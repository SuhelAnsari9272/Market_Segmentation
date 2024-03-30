import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from skearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','pca_transformer.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_pca_transformer_obj(self):
        '''
        This function is responsible for PCA implementation
        '''
        try:
            pca = PCA(n_components =2)
            return pca
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,data_path):
        try:
            
            logging.info('Obtaining PCA object')
            pca_obj = self.get_pca_transformer_obj()

            pipeline = Pipeline([
                ('scaler',StandardScaler()),
                ('pca',pca_obj)
            ])

            preprocessed_data = pipeline.fit_transform(df)

            logging.info('Saving PCA object')

            save_object(
                file_path = self.data_transformation_config.pca_obj_file_path,
                obj = pca_obj
            )

            return preprocessed_data, self.data_transformation_config.pca_obj_file_path
        
        except Exception as e:
            raise CustomException(e,sys)