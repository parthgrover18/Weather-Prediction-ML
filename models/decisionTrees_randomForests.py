import opendatasets as od
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, export_text
import warnings
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')


raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')
raw_df.dropna(subset=['RainTomorrow'], inplace=True)

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy() 

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()





# print(test_inputs[numeric_cols].isna().sum())
 
imputer = SimpleImputer(strategy='mean').fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# print(val_inputs.describe().loc[['max','min']])

scaler = MinMaxScaler().fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])



encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

 
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


model = DecisionTreeClassifier(random_state=42)


model.fit(X_train, train_targets)


train_preds = model.predict(X_train)
# print(train_preds)

# print(accuracy_score(train_preds, train_targets))







# def max_depth_error(md):
#     model = DecisionTreeClassifier(max_depth=md, random_state=42)
#     model.fit(X_train, train_targets)
#     train_error = 1 - model.score(X_train, train_targets)
#     val_error = 1 - model.score(X_val, val_targets)
#     return {'Max Depth': md, 'Training Error': train_error, 'Validation Error': val_error}

# md_errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])

# print(md_errors_df)

val_dummy = 1
def max_leaf_node_error(mfn):
    global val_dummy
    model = DecisionTreeClassifier(max_leaf_nodes=mfn, random_state=42)
    model.fit(X_train, train_targets)
    train_error = 1 - model.score(X_train, train_targets)
    val_error = 1 - model.score(X_val, val_targets)
    if val_error < val_dummy:
        val_dummy = val_error
    return {'Max Leaf Node': mfn,'Training Error': train_error, 'Validation Error': val_error}

mfn_errors_df = pd.DataFrame([max_leaf_node_error(mfn) for mfn in range(2,201)])
print(val_dummy)
mfn_errors_df.to_csv('mfn_df.csv')

# 143


















def visualisation():
    px.histogram(pd.to_datetime(raw_df.Date).dt.year, title='No. of Rows per Year', labels={'value': 'Year', 'count': 'Count'}).update_layout(bargap=0.3).show()
    
    # graph 1
    px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday').show()

    # graph 2
    px.histogram(raw_df, x='Temp3pm', title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow').show()

    # graph 3
    px.histogram(raw_df, x='RainTomorrow', color='RainToday', title='Rain Tomorrow vs. Rain Today').show()

    # graph 4
    px.scatter(raw_df.sample(2000), title='Min Temp. vs Max Temp.', x='MinTemp', y='MaxTemp', color='RainToday').show()

    # graph 5
    px.scatter(raw_df.sample(2000), title='Temp (3 pm) vs. Humidity (3 pm)', x='Temp3pm', y='Humidity3pm', color='RainTomorrow').show()
    
    # graph 6 - Bar chart of top feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
    px.bar(importance_df, x='Feature', y='Importance', title='Top 20 Feature Importances in Decision Tree').show()

