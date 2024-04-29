import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')


model = joblib.load('ChurnModell.pkl')


# Define app title and author
st.title("ChurnSight: Customer Churn Prediction & Analysis")
st.markdown("By Tumininu")

# Add space between title and author
st.markdown("<br>", unsafe_allow_html=True)

# # Add some text or description about the app

# # Add any other components you want, such as inputs, buttons, or plots

st.image('2926567-removebg-preview.png')

st.header('Project Background Information', divider = True)
st.write("The project ChurnSight: Customer Churn Prediction & Analysis aims to develop a predictive model to forecast customer churn for businesses. By analyzing historical data and identifying key factors influencing churn, ChurnSight provides insights to help businesses proactively retain customers and optimize their strategies for long-term growth and success.")


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('Churn_Modelling.csv')
st.dataframe(data.drop(['RowNumber', 'CustomerId'], axis=1))


st.sidebar.image('3999989-removebg-preview.png', caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# # Apply space in the sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# # Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
Credit = st.sidebar.selectbox('CreditScore', data['CreditScore'].unique())
salary = st.sidebar.number_input('EstimatedSalary', data['EstimatedSalary'].min(), data['EstimatedSalary'].max())
bal = st.sidebar.number_input('Balance', data['Balance'].min(), data['Balance'].max())
age = st.sidebar.number_input('Age', data['Age'].min(), data['Age'].max())
tenure = st.sidebar.number_input('Tenure', data['Tenure'].min(), data['Tenure'].max())
geo = st.sidebar.selectbox('Geography', data['Geography'].unique())
prod = st.sidebar.number_input('NumOfProducts', data['NumOfProducts'].min(), data['NumOfProducts'].max())
gen = st.sidebar.selectbox('Gender', data['Gender'].unique())
active = st.sidebar.selectbox('IsActiveMember', data['IsActiveMember'].unique())



# # display the users input
input_var = pd.DataFrame()
input_var['CreditScore'] = [Credit]
input_var['EstimatedSalary'] = [salary]
input_var['Balance'] = [bal]
input_var['Age'] = [age]
input_var['Tenure'] = [tenure]
input_var['Geography'] = [geo]
input_var['NumOfProducts'] = [prod]
input_var['Gender'] = [gen]
input_var['IsActiveMember'] = [active]


st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

salary = joblib.load('EstimatedSalary_scaler.pkl')
bal = joblib.load('Balance_scaler.pkl')
gen = joblib.load('Gender_encoder.pkl')
geo = joblib.load('Geography_encoder.pkl')

# # transform the users input with the imported scalers 
input_var['EstimatedSalary'] = salary.transform(input_var[['EstimatedSalary']])
input_var['Balance'] =  bal.transform(input_var[['Balance']])
input_var['Gender'] = gen.transform(input_var[['Gender']])
input_var['Geography'] = geo.transform(input_var[['Geography']])


model = joblib.load('ChurnModell.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict Churn'):
    if predicted == 0:
        st.error('Customer Has CHURNED')
    else:
        st.success('Customer Is With Us')