# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling and MLOps
# MAGIC * Considering historical data, build a forecasting model that infers net bike change at your station by the hour
# MAGIC * Register your model at the staging and production level within the Databricks model registry.
# MAGIC * Store artifacts in each MLflow experiment run to be used by the application to retrieve the staging and production models.
# MAGIC * Tune the model hyperparameters using the MLflow experiments and hyperparameter scaling (spark trials, hyper-opts).
# MAGIC * **YOU MUST NAME YOUR GROUP MODEL IN THE MLFLOW REGISTRY WITH YOUR GROUP’S UNIQUE NAME. SEE GROUP_MODEL_NAME GLOBAL VARIABLE.**

# COMMAND ----------

# Import libraries
import mlflow
import json
import pandas as pd
import numpy as np
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Hyperparameter tuning
import itertools

# COMMAND ----------

# Read the data from the silver table
SOURCE_DATA = spark.sql("SELECT * FROM train_bike_weather_netChange_s").toPandas()
ARTIFACT_PATH = "G10-model"
np.random.seed(12345)

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# Rename columns to match Prophet's expected format
SOURCE_DATA = SOURCE_DATA.rename(columns={'dates': 'ds', 'net_change': 'y'})

# Visualize data using seaborn
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=SOURCE_DATA['ds'], y=SOURCE_DATA['y'])
plt.legend(['Net Bike Change Data'])

# COMMAND ----------

# Initialize and fit the model
model = Prophet()
model.fit(SOURCE_DATA)

# Make hourly predictions for the next 4 hours
future = model.make_future_dataframe(periods=4, freq='H')
forecast = model.predict(future)

# Visualize the predictions
fig = model.plot(forecast)
plt.title('Hourly Net Bike Change Forecast')
plt.show()


# COMMAND ----------

#Defining a function that trains a Prophet model with given hyperparameters and logs the metrics and artifacts to MLflow
import tempfile
import os
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

def train_prophet_model(df, changepoint_prior_scale, seasonality_prior_scale, experiment_id):
    with mlflow.start_run(experiment_id=experiment_id):
        # Initialize and fit the model
        model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
        model.fit(df)

        # Cross-validation
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
        df_p = performance_metrics(df_cv, rolling_window=1)

        # Log metrics
        mlflow.log_metric("mape", df_p['mape'].mean())
        mlflow.log_metric("rmse", df_p['rmse'].mean())

        # Log hyperparameters
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)

        # Log model artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = os.path.join(temp_dir, "model.pkl")
            model.serialize(model_file)
            mlflow.log_artifact(model_file, "model")
            
        # Return the run ID
        return mlflow.active_run().info.run_id


# COMMAND ----------

# Run a hyperparameter search using MLflow experiments

# Set your MLflow experiment ID
experiment_name = "xxxx"
experiment_id = mlflow.create_experiment(experiment_name)
print(f"Experiment created with name '{experiment_name}' and ID '{experiment_id}'")

EXPERIMENT_ID = experiment_id

# Define the hyperparameter search space
changepoint_prior_scales = [0.001, 0.01, 0.1]
seasonality_prior_scales = [1, 5, 10]

# Run the hyperparameter search
for changepoint_prior_scale, seasonality_prior_scale in itertools.product(changepoint_prior_scales, seasonality_prior_scales):
    run_id = train_prophet_model(SOURCE_DATA, changepoint_prior_scale, seasonality_prior_scale, EXPERIMENT_ID)
    print(f"Finished run with changepoint_prior_scale={changepoint_prior_scale}, seasonality_prior_scale={seasonality_prior_scale}, run_id={run_id}")


# COMMAND ----------

# Get the best run based on the lowest MAPE
best_run = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["metrics.mape"]).iloc[0]

# Register the best model to the Databricks Model Registry
model_uri = f"runs:/{best_run.run_id}/model"
model_name = "GROUP_MODEL_NAME"
mlflow.register_model(model_uri, model_name)

# Promote the registered model to staging and production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)


# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
