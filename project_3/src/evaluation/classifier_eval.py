import os
import sys
import structlog
from utill.utils import load_config
import joblib 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
logger = structlog.getLogger()



class ModelEvaluation:
    def __init__(self, model, X, y, n_splits=5):

        self.model= model
        self.X = X
        self.y = y
        self.n_splits = n_splits

        pass

    def cross_val_evaluate(self):
        logger.info("Evaluation initialized")

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=load_config().get('random_state'))
        scores = cross_val_score(self.model,
                                 self.X,
                                 self.y,
                                 cv=skf,
                                 scoring=load_config().get('cross_val_metric'))
        return scores
    
    def roc_auc_scorer(self, model, X, y):
        y_pred = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred)
    
    @staticmethod
    def evaluate_predictions(y_true, y_pred_proba):
        logger.info('Initilized Model Validation')
        return roc_auc_score(y_true, y_pred_proba)