import cupy as cp
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import statistics as st
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression
from plotly.subplots import make_subplots
from scipy.stats import weibull_min
from sklearn import linear_model
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

df1 = pd.read_csv("Trend01_20221101.csv")
df2 = pd.read_csv("Trend01_20221102.csv")
df3 = pd.read_csv("Trend01_20221103.csv")
df4 = pd.read_csv("Trend01_20221104.csv")
df5 = pd.read_csv("Trend01_20221105.csv")
df6 = pd.read_csv("Trend01_20221106.csv")
df7 = pd.read_csv("Trend01_20221107.csv")

df8  = pd.read_csv("Trend01_20221108.csv")
df9  = pd.read_csv("Trend01_20221109.csv")
df10 = pd.read_csv("Trend01_20221110.csv")
df11 = pd.read_csv("Trend01_20221111.csv")
df12 = pd.read_csv("Trend01_20221112.csv")
df13 = pd.read_csv("Trend01_20221113.csv")
df14 = pd.read_csv("Trend01_20221114.csv")

df = pd.concat([df1, df2, df3, df4, df5, df6, df7,
                df8, df9, df10, df11, df12, df13, df14])

time = df['F01'].to_numpy()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

features = ['PVI_T2097', 'PVI_T2052', 'PVI_T2031']
target = 'PVI_T2031'

dtrain = xgb.DMatrix(train_df[features], label=train_df[target])
dtest = xgb.DMatrix(test_df[features], label=test_df[target])

#optuna optimisation loop
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'eta': trial.suggest_uniform('eta', 0.05, 0.3),
    }

    #xgboost train
    model = xgb.train(params, dtrain, num_boost_round=100)
    predictions = model.predict(dtest)
    actuals = test_df[target].to_numpy()
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)


# Use the best hyperparameters from Optuna to train a final model on all the training data
best_params = study.best_params
best_model = xgb.train(best_params, xgb.DMatrix(df[features], label=df[target]), num_boost_round=100)


dat = xgb.DMatrix(df[features], label=df[target])
# Use the final model for prediction on new data
new_data = dat # data that need to make predictions on
# new_dmatrix = xgb.DMatrix(new_data)
new_predictions = best_model.predict(dat)


fig = px.line(x = time, y = new_predictions)
fig.show()


