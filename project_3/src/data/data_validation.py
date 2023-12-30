
import pandas as pd
import os
import sys
import structlog
import pandera
from pandera import Check, Column, DataFrameSchema
from utill.utils import load_config

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
logger = structlog.getLogger()


class DataValidation:
    def __init__(self) -> None:
        self.columns_to_use = load_config().get('columns_to_use')

    def check_shape_data(self, dataframe: pd.DataFrame) -> bool:
        
        try:
            print('Initial Validation')
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            print(f'Validadtion Failed: {e}')
            return False
        
    def check_columns(self, dataframe: pd.DataFrame) -> bool:
        schema = DataFrameSchema(
            {
                # Check if it is integer and if its values are between 0 and 1. See if it is greater than zero to confirm that are no missing values, If there is a problem, warn us (coerce)
                'target': Column(int, Check.isin([0, 1]), Check(lambda x: x > 0), coerce=True),
                'TaxaDeUtilizacaoDeLinhasNaoGarantidas': Column(float, nullable=True), 
                'Idade': Column(int, nullable=True),
                'NumeroDeVezes30-59DiasAtrasoNaoPior': Column(int, nullable=True), 
                'TaxaDeEndividamento': Column(float, nullable=True),
                'RendaMensal': Column(float, nullable=True), 
                'NumeroDeLinhasDeCreditoEEmprestimosAbertos': Column(int, nullable=True),
                'NumeroDeVezes90DiasAtraso': Column(int, nullable=True), 
                'NumeroDeEmprestimosOuLinhasImobiliarias': Column(int, nullable=True),
                'NumeroDeVezes60-89DiasAtrasoNaoPior': Column(int, nullable=True), 
                'NumeroDeDependentes': Column(float, nullable=True)
                
            }
        )

        try:
            schema.validate(dataframe)
            logger.info('Validation columns passed')
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error('Validadtion columns failed')
            pandera.display(exc.failure_cases)
        return False
    
    
    def run(self, dataframe: pd.DataFrame) -> bool:
        if self.check_shape_data(dataframe) and self.check_columns(dataframe):
            logger.info('Successful Validation.')
            return True
        else:
            logger.error('Validation Failed')
            return False