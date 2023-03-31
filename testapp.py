import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a SVM model using all 30 features
svm_model = SVC(kernel='linear', C=1, probability=True)
svm_model.fit(X_train_scaled, y_train)

# Define the Streamlit app
st.title('Wisconsin Breast Cancer Prediction App')

# Define sliders for feature input
feature_names = data.feature_names
slider_dict = {}
for feature in feature_names:
    slider_dict[feature] = st.slider(f'{feature}', float(data.data[:, feature_names.tolist().index(feature)].min()), float(data.data[:, feature_names.tolist().index(feature)].max()), float(data.data[:, feature_names.tolist().index(feature)].mean()))

# Define a button to make predictions
button = st.button('Make prediction')

if button:
    # Make a prediction using the SVM model
    prediction_input = pd.DataFrame([slider_dict])
    prediction_input_scaled = scaler.transform(prediction_input)
    prediction = svm_model.predict(prediction_input_scaled)

    # Display the prediction
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.write('The tumor is benign.')
    else:
        st.write('The tumor is malignant.')
