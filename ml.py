import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_data(dir_name, feature):
    suffix = '.csv'
    filepath = Path(dir_name, feature).with_suffix(suffix)
    csv_data = pd.read_csv(filepath)
    return csv_data


dir_from = './data-features-extracted'
x_columns = ['f(t)', 'f(t-1)', 'f(t-2)', 'f(t)-f(t-1)', 'f(t)-f(t-2)']
y_columns = ['flag']

data = read_data(dir_from, 'angle')
x, y = data['f(t)'], data[y_columns]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=7)
print(xtest)
print(ytest)

xgbc = XGBClassifier(learning_rate=0.1, n_estimators=1000, early_stopping_rounds=10, max_depth=15, use_label_encoder=False)

xgbc.fit(xtrain, ytrain)

y_pred = xgbc.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
print(accuracy)
