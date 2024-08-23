# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#####################
#### Nick Lunkov and Connor Cassedy Final Project code
#####################

#####################
#### Importing libraries/Getting Data
#####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Setting to make looking at data easier
pd.options.display.max_columns = None




# Importing ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

# Load the dataset
cali = fetch_california_housing(as_frame = True)
cali_df = cali.frame

#####################
#### EDA of California Dataset
#####################
#Looking at top 10 records
cali_df.head(10)

#Looking at type of column and null counts
cali_df.info()

#Getting basic statistics of columns
cali_df.describe()

#Histograms of all variables in dataset
histogram = cali_df.hist(figsize = (10,10))

#Correlation calculation and heatmap
cali_df.corr()
ax = plt.axes()
sns.heatmap(cali_df.corr(), linewidths=1, linecolor='black', ax = ax)
ax.set_title('Correlation Heatmap of Cali Dataset Variables')

#Scatterplot of housing values based on latitude and longitude
ax = plt.axes()
sns.scatterplot(data = cali_df, x = "Longitude", y = "Latitude", size = "MedHouseVal", hue = "MedHouseVal", alpha=0.5)
ax.set_title('Latitude/Longitude Scatterplot based on Median House Value')

#Counting outliers per column
cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

#Splitting data between target and input variables
y_target = cali_df['MedHouseVal']
X_features = cali_df.drop(['MedHouseVal'], axis = 1)




#####################
#### Standardizing the dataset
#####################

#Splitting data into train and test
#Normaling the target feature by taking the log
y_target_log = np.log1p(y_target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size = 0.25, random_state = 2)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#####################
#### Setting metrics, going to use basic R^2 as the test statistic
#####################
from sklearn.model_selection import cross_val_score

def calculate_eval_metric(model, X, y, cv = 3):
    scores = cross_val_score(model, X, y, cv = cv, scoring = 'r2')
    print("Evaluation score on 3 cross-validation sets : ", scores)
    print("Average R squared score : ", scores.mean())
    return scores.mean()

#####################
#### Performing basic XGBoost
#####################
import xgboost as XGB

scores = {}

#Creating basic model
xgb1 = XGB.XGBRegressor()
scores["xgb1"] = calculate_eval_metric(xgb1, X_train, y_train)

xgb_basic = xgb1.fit(X_train, y_train)
xgb_basic.score(X_test, y_test)

#####################
#### Performing hyperparameter tuning using GridSearchCV
#####################
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats

params = {
    'max_depth': [1, 3, 5, 7, 9],
    'learning_rate': [0.1, 0.01, 0.001, .0001],
    'subsample': [.01,.03, 0.5, 0.7, 1]
}

grid_search = GridSearchCV(xgb_basic, params, cv=3, scoring='r2')

grid_search.fit(X_train, y_train)

print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)




#####################
#### Performing hyperparameter tuning using Hyperopt
#####################
import hyperopt.hp as hp
from hyperopt import fmin, tpe, Trials, STATUS_OK


def objective(params):
    model = XGB.XGBRegressor(**params, random_state=2)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    mean_score = scores.mean()
    return {'loss': -mean_score, 'status': STATUS_OK, 'scores': scores}

space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int))
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

print("Best hyperparameters found by Hyperopt: ", best)

# Fit and score the final model using the best hyperparameters
best_params = {
    'max_depth': best['max_depth'],
    'learning_rate': best['learning_rate'],
    'subsample': best['subsample'],
    'n_estimators': best['n_estimators']
}

best_model = XGB.XGBRegressor(**best_params, random_state=2)
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)
print("Final model score on test set: ", final_score)

#####################
#### Evaluation of Hyperopt Process
#####################
# Best score evolution
best_scores = [-trial['result']['loss'] for trial in trials.trials]
plt.plot(best_scores, label='Best score evolution')
plt.xlabel('Trials')
plt.ylabel('R^2 score')
plt.title('Best Score Evolution during Hyperopt')
plt.legend()
plt.show()

# Average score progression
average_scores = [np.mean([trial['result']['scores'] for trial in trials.trials[:i+1]]) for i in range(len(trials.trials))]
plt.plot(average_scores, label='Average score progression')
plt.xlabel('Trials')
plt.ylabel('Average R^2 score')
plt.title('Average Score Progression during Hyperopt')
plt.legend()
plt.show()

# Hyperparameter convergence
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot([trial['misc']['vals']['max_depth'][0] for trial in trials.trials], label='max_depth')
axes[0, 0].set_xlabel('Trials')
axes[0, 0].set_ylabel('Max Depth')
axes[0, 0].set_title('Convergence of max_depth')
axes[0, 0].legend()

axes[0, 1].plot([trial['misc']['vals']['learning_rate'][0] for trial in trials.trials], label='learning_rate')
axes[0, 1].set_xlabel('Trials')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Convergence of learning_rate')
axes[0, 1].legend()

axes[1, 0].plot([trial['misc']['vals']['subsample'][0] for trial in trials.trials], label='subsample')
axes[1, 0].set_xlabel('Trials')
axes[1, 0].set_ylabel('Subsample')
axes[1, 0].set_title('Convergence of subsample')
axes[1, 0].legend()

axes[1, 1].plot([trial['misc']['vals']['n_estimators'][0] for trial in trials.trials], label='n_estimators')
axes[1, 1].set_xlabel('Trials')
axes[1, 1].set_ylabel('Number of Estimators')
axes[1, 1].set_title('Convergence of n_estimators')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

#####################
#### Performing hyperparameter tuning using Optuna
#####################
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    model = XGB.XGBRegressor(**params, random_state=2)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    mean_score = scores.mean()
    return mean_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters found by Optuna: ", study.best_params)

# Fit and score the final model using the best hyperparameters
best_params = study.best_params

best_model = XGB.XGBRegressor(**best_params, random_state=2)
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)
print("Final model score on test set: ", final_score)

#####################
#### Evaluation of Optuna Process
#####################
# Best score evolution
best_scores = [trial.value for trial in study.trials]
plt.plot(best_scores, label='Best score evolution')
plt.xlabel('Trials')
plt.ylabel('R^2 score')
plt.title('Best Score Evolution during Optuna')
plt.legend()
plt.show()

# Average score progression
average_scores = [np.mean([trial.value for trial in study.trials[:i+1]]) for i in range(len(study.trials))]
plt.plot(average_scores, label='Average score progression')
plt.xlabel('Trials')
plt.ylabel('Average R^2 score')
plt.title('Average Score Progression during Optuna')
plt.legend()
plt.show()

# Hyperparameter convergence
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot([trial.params['max_depth'] for trial in study.trials], label='max_depth')
axes[0, 0].set_xlabel('Trials')
axes[0, 0].set_ylabel('Max Depth')
axes[0, 0].set_title('Convergence of max_depth')
axes[0, 0].legend()

axes[0, 1].plot([trial.params['learning_rate'] for trial in study.trials], label='learning_rate')
axes[0, 1].set_xlabel('Trials')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Convergence of learning_rate')
axes[0, 1].legend()

axes[1, 0].plot([trial.params['subsample'] for trial in study.trials], label='subsample')
axes[1, 0].set_xlabel('Trials')
axes[1, 0].set_ylabel('Subsample')
axes[1, 0].set_title('Convergence of subsample')
axes[1, 0].legend()

axes[1, 1].plot([trial.params['n_estimators'] for trial in study.trials], label='n_estimators')
axes[1, 1].set_xlabel('Trials')
axes[1, 1].set_ylabel('Number of Estimators')
axes[1, 1].set_title('Convergence of n_estimators')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
