import pandas as pd
import os
import sys
import structlog
from utill.utils import load_config
import joblib 

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import MeanMedianImputer
from evaluation.classifier_eval import ModelEvaluation
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler, StandardScaler
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
logger = structlog.getLogger()


mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('prob_loan')

class TrainModels:

    def __init__(self, X_data: pd.DataFrame, y_data:pd.DataFrame):
        self.X_data = X_data
        self.y_data = y_data
        self.model_path = load_config().get('model_path')

    def get_best_model(self):
        logger.info('Selecting best model on mflow')
        df_mlflow = mlflow.search_runs(filter_string='metrics.valid_roc_auc < 1').sort_values('metrics.valid_roc_auc', ascending=False)
        run_id  = df_mlflow.loc[df_mlflow['metrics.valid_roc_auc'].idxmax()].run_id

        df_best_params = df_mlflow.query(f'run_id == "{run_id}"').filter(like='params')

        best_roc_auc = df_mlflow.query(f'run_id == "{run_id}"')['metrics.valid_roc_auc']
        return best_roc_auc, df_best_params
    
    def run(self):
        _, best_params_df = self.get_best_model()

        logger.info(f'Initializing Model Training: {self.model_path}')

        with mlflow.start_run(run_name='final_model'):
            mlflow.set_tag('model_name', self.model_path)


            model = LogisticRegression(warm_start=eval(best_params_df['params.warm_start'].values[0]),
                                       class_weight=eval(best_params_df['params.class_weight'].values[0]),
                                       tol=float(best_params_df['params.tol'].values[0]),
                                       max_iter=int(best_params_df['params.max_iter'].values[0]),
                                       solver=best_params_df['params.solver'].values[0],
                                       multi_class=best_params_df['params.multi_class'].values[0],
                                       C=float(best_params_df['params.C'].values[0]),
                                       fit_intercept=eval(best_params_df['params.fit_intercept'].values[0])
                                       )
            
            pipe = Pipeline([
                 ('imputer', eval(best_params_df['params.imputer'].values[0])), 
                 ('discretizer', eval(best_params_df['params.discretizer'].values[0])),
                 ('scaler', eval(best_params_df['params.scaler'].values[0])),
                 ('model', model)
                 ])
            
            pipe.fit(self.X_data, self.y_data)

            # log evaluation metrics
            y_val_preds = pipe.predict_proba(self.X_data)
            model_eval = ModelEvaluation(model,
                                         self.X_data,
                                         self.y_data)
            val_roc_auc = model_eval.evaluate_predictions(self.y_data, y_val_preds[:,1])
            mlflow.log_metric('valid_roc_auc', val_roc_auc)

            # register model
            mlflow.sklearn.log_model(pipe,
                                     'final_model',
                                     pyfunc_predict_fn='predict_proba',
                                     input_example=self.X_data.iloc[[0]],
                                     registered_model_name='final_model')

    #metódo para deixar o método privado da classe
    def _save_model(self, model_fitted):
        os.path.abspath(self.model_path)
        joblib.dump(model_fitted, self.model_path)