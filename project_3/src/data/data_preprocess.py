import pandas as pd
import os
import sys
import structlog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utill.utils import load_config

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
logger = structlog.getLogger()

class DataPreprocess:

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.trained_pipe = None

    def train(self, dataframe:pd.DataFrame):
        logger.info('Initialized Preprocessing')
        self.trained_pipe = self.pipe.fit(dataframe)

    def transform(self, dataframe: pd.DataFrame):
        if self.trained_pipe is None:
            raise ValueError('Pipeline not trained.')
        logger.info('Data Transformation with preprocess started...')
        data_processed = self.trained_pipe.transform(dataframe)
        return data_processed
    

    #def pipline(self):
    #    train_pipe = self.pipe
    #    train_pipe.fit(self.dataframe)
    #    return train_pipe
    #
#
    #def run(self):
    #    print('Preprocess initialized')
    #    trained_pipeline = self.pipline()
    #    data_preprocessed = trained_pipeline.transform(self.dataframe)
    #    return data_preprocessed