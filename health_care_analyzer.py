import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Heart Disease Analyzer", layout="wide")

# Streamlit title and introduction
st.title("Heart Disease Data Analysis")
st.markdown("""
    <style>
    .main {
        background-color: #ADD8E6;
        color: black; 
    }
    .sidebar .sidebar-content {
        background-color: #ffcccc;  /* Light red sidebar */
        color: black; /* Sets sidebar text color to black */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    blood_pressure = st.sidebar.number_input("Enter Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    age = st.sidebar.number_input("Enter Age", min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox("Select Sex", ["Male", "Female"])
    return blood_pressure, age, sex

# Get user inputs
blood_pressure, age, sex = user_input_features()

# Load data
data = pd.read_csv('/Downloads/heart.csv')

# Display dataset
st.header("Dataset Preview")
st.write(data)

# Display basic information about the dataset
st.header("Data Information")
st.write("Shape of the dataset:", data.shape)
st.write("Summary of the dataset:")
st.write(data.describe())
st.write("Data Types and Non-Null Counts:")
buffer = st.empty()  # Placeholder for better display
buffer.write(data.info())
buffer.write("Null Values:")
buffer.write(data.isnull().sum())
buffer.write("Any Duplicates:")
data_dup = data.duplicated().any()
buffer.write(data_dup)

# Drop duplicates
data = data.drop_duplicates()
buffer.write("Shape after dropping duplicates:")
buffer.write(data.shape)

# Correlation matrix
st.header("Correlation Matrix")
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap", fontsize=16, color='navy')
st.pyplot()

# Target variable counts
st.header("Target Variable Counts")
st.write(data['target'].value_counts())

# Countplot for target variable
st.header("Countplot of Target Variable")
fig, ax = plt.subplots()
sns.countplot(data['target'], ax=ax, palette="Set1")  # Using Set1 for vibrant colors
ax.set_title("Count of Heart Disease Cases", fontsize=16, color='darkred')
st.pyplot(fig)

# Countplot for sex variable
st.header("Countplot of Sex Variable")
fig, ax = plt.subplots()
sns.countplot(data['sex'], ax=ax, palette="Set1")
ax.set_xticklabels(['Female', 'Male'])
ax.set_title("Gender Distribution", fontsize=16, color='darkred')
st.pyplot(fig)

# Countplot for sex vs target
st.header("Countplot of Sex by Target")
fig, ax = plt.subplots()
sns.countplot(x='sex', hue='target', data=data, ax=ax, palette="Set1")
ax.set_xticklabels(['Female', 'Male'])
ax.set_title("Gender vs Heart Disease", fontsize=16, color='darkred')
st.pyplot(fig)

# Distribution plot for age
st.header("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(data['age'], kde=True, ax=ax, color='orange', bins=20)
ax.set_title("Age Distribution with KDE", fontsize=16, color='darkred')
st.pyplot(fig)

# Countplot for chest pain type
st.header("Countplot of Chest Pain Type")
fig, ax = plt.subplots()
sns.countplot(data['cp'], ax=ax, palette="Set1")
ax.set_title("Chest Pain Types", fontsize=16, color='darkred')
st.pyplot(fig)

# Countplot for fasting blood sugar vs target
st.header("Countplot of Fasting Blood Sugar by Target")
fig, ax = plt.subplots()
sns.countplot(x='fbs', hue='target', data=data, ax=ax, palette="Set1")
ax.set_title("Fasting Blood Sugar vs Heart Disease", fontsize=16, color='darkred')
st.pyplot(fig)

# Prediction Section
st.header("Heart Disease Prediction Based on User Input")

# Checkbox for making a prediction
predict = st.sidebar.checkbox("Predict Heart Disease")

if predict:
    # Simple rule-based prediction (replace with your model logic)
    if blood_pressure > 140:  # This is just an example condition
        st.success("High blood pressure detected: Possible heart disease risk!", icon="⚠️")
    else:
        st.success("Blood pressure is within a normal range: Lower risk of heart disease!", icon="✅")

    # Provide additional feedback
    st.write(f"**Input Values:**\n- Blood Pressure: {blood_pressure} mm Hg\n- Age: {age}\n- Sex: {sex}")
    st.info("This is a simple prediction. For more accurate results, consider using a trained machine learning model.")
