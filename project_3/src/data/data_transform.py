import pandas as pd
import os
import sys
import structlog
from sklearn.model_selection import train_test_split
from utill.utils import load_config

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
logger = structlog.getLogger()



class DataTransformation:

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.target_name = load_config().get('target_name')

    def train_test_splitting(self):
        
        x = self.dataframe.drop(self.target_name, axis=1)
        y = self.dataframe[self.target_name]

        X_train, X_valid, y_train, y_valid = train_test_split(x, y, 
                                                              test_size=load_config().get('test_size'), stratify=y, random_state=load_config().get('random_state'))

        return X_train, X_valid, y_train, y_valid