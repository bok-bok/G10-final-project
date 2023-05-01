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

spark.sql("SELECT * FROM train_bike_weather_netChange_s").toPandas()

# COMMAND ----------

# Read the data from the silver table
SOURCE_DATA = spark.sql("SELECT * FROM train_bike_weather_netChange_s").toPandas()
ARTIFACT_PATH = "G10-model"
np.random.seed(12345)

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# Rename columns to match Prophet's expected format
SOURCE_DATA = SOURCE_DATA.rename(columns={'dates': 'ts', 'net_change': 'y'})

# Visualize data using seaborn
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=SOURCE_DATA['ts'], y=SOURCE_DATA['y'])
plt.legend(['Net Bike Change Data'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Baseline Model and record the performance

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data preprocessing

# COMMAND ----------

SOURCE_DATA = spark.sql("SELECT * FROM train_bike_weather_netChange_s").toPandas()
SOURCE_DATA.head()

# COMMAND ----------

SOURCE_DATA['feels_like'].fillna(0, inplace=True)
SOURCE_DATA['rain_1h'].fillna(0, inplace=True)
SOURCE_DATA['holiday'] = SOURCE_DATA['holiday'].apply(lambda x: 1 if x == "true" else 0)
# Rename net_change column to 'y'
SOURCE_DATA.rename(columns={'net_change': 'y'}, inplace=True)
SOURCE_DATA = SOURCE_DATA.drop(['dayofweek', 'description'], axis=1)
SOURCE_DATA = SOURCE_DATA.rename(columns={'ts': 'ds'})

SOURCE_DATA.head()

# COMMAND ----------

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Initiate the model
baseline_model = Prophet()

# Add additional regressors
baseline_model.add_regressor('year')
baseline_model.add_regressor('month')
baseline_model.add_regressor('dayofmonth')
baseline_model.add_regressor('hour')
baseline_model.add_regressor('feels_like')
baseline_model.add_regressor('rain_1h')
baseline_model.add_regressor('holiday')


# Fit the model on the training dataset
baseline_model.fit(SOURCE_DATA)

# Cross validation
baseline_model_cv = cross_validation(baseline_model, initial='30 days', period='7 days', horizon='4 hours',parallel="threads")

# Model performance metrics
baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
baseline_model_p.head()

# Get the performance value
print(f"MAE of baseline model: {baseline_model_p['mae'].values[0]}")



# COMMAND ----------



import matplotlib.pyplot as plt

# Make hourly predictions for the next 4 hours
future = baseline_model.make_future_dataframe(periods=4, freq='H')

# Add the additional regressors to the future dataframe
future['year'] = future['ds'].dt.year
future['month'] = future['ds'].dt.month
future['dayofmonth'] = future['ds'].dt.day
future['hour'] = future['ds'].dt.hour
future['holiday'] = SOURCE_DATA['holiday'].astype(int)
future['feels_like'] = SOURCE_DATA['feels_like']
future['rain_1h'] = SOURCE_DATA['rain_1h']

forecast = baseline_model.predict(future)

# Visualize the predictions
fig = baseline_model.plot(forecast)
plt.title('Hourly Net Bike Change Forecast')
plt.show()

# COMMAND ----------



# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
