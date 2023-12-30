
import pandas as pd
import os
import sys
import structlog
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utill.utils import load_config

logger = structlog.getLogger()


class DataLoad:
    """Class data load"""
    def __init__(self) -> None:
        pass

    def load_data(self, dataname: str) -> pd.DataFrame:
        """Load data from dataset name.
        
        Args:
            dataname (str): Datase name to be loaded.
            
        Returns:
        Raises:
        """
        logger.info(f'Starting data load: {dataname}')  
        dataset = load_config().get(dataname)

        try:
            dataset = load_config().get(dataname)
            if dataset is None:
                raise ValueError(f'Error: The name of the dataset is wrong: {dataset}.')

            loaded_data = pd.read_csv('../data/raw/train.csv')
            return loaded_data
        except ValueError as ve:
            logger.error(str(ve))
        except Exception as e:
            logger.error(f'Unexpected Error: {str(e)}')
        
