import streamlit as st
import pandas as pd
import os
import joblib
import json
from utils import transform_data
import matplotlib.pyplot as plt 
import seaborn as sns

# Write info on the display
st.title('Churn Prediction App')
st.write('My first streamlit write from python')

st.sidebar.title('Chunr Assets')

# load the schema
with open(os.path.join('schema.json'), 'r') as f: # reload the schema (testing it)
    schema = json.load(f)
#st.write(schema)

column_order_in = list(schema['col_info'].keys())[1:-1] # extract column order
column_order_out = list(schema['transformed_columns'])  # get the output order
#st.write(column_order_in)

st.sidebar.info('Customer Churn Features for Prediction')


options = {}
for col, col_properties in schema['col_info'].items():
    if col == 'id':
        continue
    if col == 'churn':
        continue
    if col_properties['dtype'] == 'int64' or col_properties['dtype'] == 'float64':
        min_, max_ = col_properties['values']
        dtype = col_properties['dtype']

        mean_ =  (min_ + max_)//2
#        if dtype=='int64':
        mean_ = int(mean_)
        options[col] = st.sidebar.slider(col, int(min_), int(max_), value=mean_)
    if col_properties['dtype']=='object':
        options[col] = st.sidebar.selectbox(col, col_properties['values'])


# Load the model
with open(os.path.join('..', 'models', 'experiment_1', 'xg.joblib'), 'rb') as f:
    model = joblib.load(f)

# Load the encoder
with open(os.path.join('..', 'models', 'experiment_1', 'ohe.joblib'), 'rb') as f:
    ohe = joblib.load(f)

# mean eve minutes
mev = 200.29

if st.button('Predict'):
    data = pd.Series(options).to_frame().T
    data = data[column_order_in]

    # check data types
    for col, col_properties in schema['col_info'].items():
        if col=='id':
            continue
        if col=='churn':
            continue
        dtype = col_properties['dtype']
        data[col] = data[col].astype(dtype)

    data = transform_data(data, column_order_out, mev, ohe)

    # Render Predictions
    pred = model.predict(data)
    st.write('Predicted Outcome')
    st.write(pred)
    st.write('Client details')
    st.write(options)

    # Save Historical data
    try:
        hist = pd.Series(options).to_frame().T
        hist['prediction'] = pred
        if os.path.isfile('historical.csv'):
            hist.to_csv('historical.csv', mode='a', header=False, index=False)
        else:
            hist.to_csv('historical.csv', header=True, index=False)
    except Exception as e:
        st.write(e)
        pass

st.header('Historical Outcomes')
if os.path.isfile('historical.csv'):
    hist = pd.read_csv('historical.csv')
    st.dataframe(hist)
    fig, ax = plt.subplots()
    sns.countplot(x='prediction', data=hist, ax=ax).set_title('Historical Predictions')
    st.pyplot(fig)
else:
    st.write('No historical data')