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
SOURCE_DATA = ("")
ARTIFACT_PATH = "G10-model"
np.random.seed(12345)

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


sales_data = pd.read_csv(SOURCE_DATA)
print(f"{len(sales_data)} months of sales data loaded ({round(len(sales_data)/12,2)} years)")

# Visualize data using seaborn
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=sales_data['ds'], y=sales_data['y'])
plt.legend(['Sales Data'])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
