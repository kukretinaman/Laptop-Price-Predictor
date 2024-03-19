import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import sklearn

# print(sklearn.__version__)

file = open('laptop_price_prediction.pkl', 'rb')
rf = pickle.load(file)
file.close()

data = pd.read_csv('trained_data (1).csv')
data['IPS'].unique()
st.title('Laptop Price Predictor')

# fitted pipe
file1 = open('pipe.pkl', 'rb')
pipe = pickle.load(file1)
file1.close()

#company
company = st.selectbox('Brand', data['Company'].unique())

#Types of laptop
type = st.selectbox('Type', data['TypeName'].unique())

#Ram present
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

#os of the laptop
os = st.selectbox('OS', data['OpSys'].unique())

#weight of the laptop
weight = st.number_input('Weight of the laptop')

#touchscreen
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

#ips
ips = st.selectbox('IPS', ['Yes', 'No'])

#screensize
screen_size = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1880',
    '1440x900', '2304x1440', '3200x1800', '2160x1440', '2560x1440',
    '2736x1824', '2400x1600', '1920x1200'
])

#cpu
cpu = st.selectbox('CPU', data['CPU_name'].unique())

#hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

#ssd
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

#flashstorage
fs = st.selectbox('Flash Storage(in GB)', [0, 128, 256, 512, 1024, 2048])

#gpu
gpu = st.selectbox('GPU(inGB)', data['Gpu_brand'].unique())

if st.button('Predict Price'):

  ppi = None

  if touchscreen == 'Yes':
    touchscreen = 1
  else:
    touchscreen = 0

  if ips == 'Yes':
    ips = 1
  else:
    ips = 0

  X_res = int(resolution.split('x')[0])
  Y_res = int(resolution.split('x')[1])

  ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

  weight = float(weight)

  # query = np.array([
  #     company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, fs,
  #     gpu, os
  # ])
  # query = query.reshape(1, 13)

  # y_pred = rf.predict(query)
  # prediction = int(np.exp(y_pred[0]))

  # st.title("Predicted price for this laptop is between: ₹" +
  #          str(prediction - 1000) + "& ₹" + str(prediction + 1000))

  # Handle categorical encoding
  query = pd.DataFrame({
      'Company': [company],
      'TypeName': [type],
      'Ram': [ram],
      'OpSys': [os],
      'Weight': [weight],
      'Touchscreen': [touchscreen],
      'IPS': [ips],
      'PPI': [ppi],
      'CPU_name': [cpu],
      'HDD': [hdd],
      'SSD': [ssd],
      'Flash_Storage': [fs],
      'Gpu_brand': [gpu]
  })

  # Transform categorical columns using the pipeline
  query_encoded = pipe.named_steps['step1'].transform(query)

  # #query_encoded converted to numpy array
  # query_encoded = query_encoded.reshape(1, 13)

  # Make prediction
  prediction = rf.predict(query_encoded)

  # Convert prediction to price
  prediction_price = int(np.exp(prediction[0]))

  st.title("Predicted price for this laptop is between: ₹" +
           str(prediction_price - 1000) + " and ₹" +
           str(prediction_price + 1000))
