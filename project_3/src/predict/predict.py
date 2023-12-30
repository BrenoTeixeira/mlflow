import requests
import json
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import structlog
import sqlite3

conn = sqlite3.connect("../../preds.db")

cursor = conn.cursor()
logger = structlog.get_logger()

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("prob_loan")


class Predict:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.endpoint = "http://127.0.0.1:8081/invocations"

    def run(self):
        logger.info("Initializing Prediction.")
        to_inference = {
            "dataframe_split": {
                "columns": self.dataframe.columns.tolist(),
                "data": self.dataframe.replace(np.nan, None).values.tolist(),
            }
        }

        response = requests.post(self.endpoint, json=to_inference)

        logger.info("Predictions Completed.")

        probabilities = np.array(json.loads(response.text).get("predictions", []))[:, 1]

        df_probs = self._results(probabilities)

        self._capture_inputs_and_predictions(to_inference, df_probs)
        logger.info("Results Saved.")

        return df_probs

    def _capture_inputs_and_predictions(self, inputs, preds):
        input_df = pd.DataFrame(
            inputs["dataframe_split"]["data"],
            columns=inputs["dataframe_split"]["columns"],
        )

        input_df["preds_prob"] = preds

        self._store_in_database(input_df)

    def _store_in_database(self, input_df):
        input_df.to_sql("predictions", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

    def _results(self, probabilities: np.array):
        df_results = pd.DataFrame()
        logger.info("Saving probabilities")
        df_results["probabilities"] = probabilities
        return df_results
