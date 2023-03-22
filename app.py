# import streamlit as st
# import pandas as pd






import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px


# Define function for page 3: Group Members
def page_group_members():
    st.title("Breast Cancer Diagnosis Prediction")
    st.title("Introduction")
    st.title("App for Group 2's Final Year Project's Data Analysis/Machine Learning Component\n\nComputer Science Department, School of Computing and Engineering Sciences, Babcock University, Ilishan-Remo, Ogun State, Nigeria.\n\nGroup 2 Members:\n\n1. Ogunsusi, Adetunji Gabriel 19/2500\n2. Okoye, Adaora Jessica 19/0686\n3. Sawo, Collins 19/1080\n\nGroup Project Supervisor: Ernest E. Onuiri, PHD\n\r")
    # Load image from URL
    url = "https://images.pexels.com/photos/5702171/pexels-photo-5702171.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    st.image(url, caption=" ")
    
# Define function for page 1: Data Set Description
def page_data_description():
    st.title("Data Set Description")
    st.write("Here is a description of our data set:")
    
    # Load data set
    df = pd.read_csv('data.csv')
    
    # Display data set summary and description
    st.subheader("Data Set Summary")
    st.write(df.describe())
    st.subheader("Data Set Description")
    st.write("This data set contains information about...")
    
    # Display data cleaning process and visualization
    st.subheader("Data Cleaning")
    st.write("Here's how we cleaned the data:")
    # Data Cleaning
    st.subheader("Data Cleaning")
    
    # Drop columns with missing values
    df = df.dropna(axis=1)
    st.write("We dropped the columns with missing values.")
    
    # Label encode the diagnosis column
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    st.write("We label encoded the diagnosis column.")
    
    # Display cleaned data set
    st.subheader("Cleaned Data Set")
    st.write(df)

    st.subheader("Data Visualization")
    st.write("Here are some visualizations of the data:")
    
    # Create plot using Plotly
    fig = px.histogram(df, x='diagnosis')

    # Set plot title and labels
    fig.update_layout(
        title='Diagnosis Counts',
        xaxis_title='Diagnosis',
        yaxis_title='Count'
    )

    # Display plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=100)
    sns.heatmap(df.corr(), ax=ax2)
    st.pyplot(fig2)
    
    # Bar plot of radius_mean and texture_mean
    st.subheader("Bar Plot of Radius Mean vs Texture Mean")
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    sns.barplot(x="radius_mean", y="texture_mean", data=df[170:180], ax=ax3)
    ax3.set_title("Radius Mean vs Texture Mean", fontsize=15)
    ax3.set_xlabel("Radius Mean")
    ax3.set_ylabel("Texture Mean")
    st.pyplot(fig3)
    
    # Line plot of concavity_mean and concave points_mean
    st.subheader("Line Plot of Concavity Mean vs Concave Mean")
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="concavity_mean", y="concave points_mean", data=df[0:400], color='green', ax=ax4)
    ax4.set_title("Concavity Mean vs Concave Mean")
    ax4.set_xlabel("Concavity Mean")
    ax4.set_ylabel("Concave Points")
    st.pyplot(fig4)


# Define function for page 2: Predictions

def load_preprocessor():
    with open('scaler.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

def load_svm_model():
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model

# Define a function to make predictions on new data

def predict_diagnosis(data):
    X_new = pd.DataFrame(data, index=[0])
    preprocessor = load_preprocessor() # load preprocessor from pickle file
    
    # extract the ColumnTransformer object and feature names
    column_transformer = preprocessor.named_transformers_['preprocessor']
    feature_names_in = column_transformer.get_feature_names_in()
    
    # check if all feature names are present in X_new
    if not set(feature_names_in).issubset(set(X_new.columns)):
        missing_features = set(feature_names_in) - set(X_new.columns)
        raise ValueError("Missing features in input data: {}".format(missing_features))
    
    X_new = preprocessor.transform(X_new)
    feature_names_out = column_transformer.get_feature_names_out() # extract feature names from ColumnTransformer
    svm_model = load_svm_model() # load SVM model from pickle file
    y_pred = svm_model.predict(X_new)
    return y_pred[0], feature_names_out

# def predict_diagnosis(data):
#     X_new = pd.DataFrame(data, index=[0])
#     preprocessor = load_preprocessor() # load preprocessor from pickle file
#     X_new = X_new.fillna(0)
#     X_new = preprocessor.transform(X_new)
#     feature_names = preprocessor.named_transformers_['preprocess'].named_steps['ct'].get_feature_names_out() 
#     svm_model = load_svm_model() # load SVM model from pickle file
#     y_pred = svm_model.predict(X_new)
#     return y_pred[0], feature_names



def page_predictions():
    st.title("Predictions")
    st.write("Here's where you can make predictions based on the data:")

    # Load data
    df = pd.read_csv('data.csv')

    st.write("Enter the values below to predict the diagnosis.")
    radius_mean = st.slider("Radius Mean", 6.98, 28.11, 16.17)
    texture_mean = st.slider("Texture Mean", 9.71, 39.28, 21.41)
    perimeter_mean = st.slider("Perimeter Mean", 43.79, 188.5, 91.97)
    area_mean = st.slider("Area Mean", 143.5, 2501.0, 654.89)
    concavity_mean = st.slider("Concavity Mean", 0.0, 0.4275, 0.1)
    concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.2012, 0.1)
    symmetry_mean = st.slider("Symmetry Mean", 0.106, 0.304, 0.179)
    fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.04996, 0.09744, 0.0628)
    radius_se = st.slider("Radius SE", 0.1115, 2.873, 0.4985)
    texture_se = st.slider("Texture SE", 0.3602, 4.885, 1.248)
    perimeter_se = st.slider("Perimeter SE", 0.757, 21.98, 2.866)
    area_se = st.slider("Area SE", 6.802, 542.2, 40.32)
    smoothness_se = st.slider("Smoothness SE", 0.001713, 0.03113, 0.00875)
    compactness_se = st.slider("Compactness SE", 0.002252, 0.1354, 0.03241)
    concavity_se = st.slider("Concavity SE", 0.0, 0.396, 0.04112)
    concave_points_se = st.slider("Concave Points SE", 0.0, 0.05279, 0.02184)
    symmetry_se = st.slider("Symmetry SE", 0.007882, 0.07895, 0.02059)
    fractal_dimension_se = st.slider("Fractal Dimension SE", 0.0008948, 0.02984, 0.006153)
    radius_worst = st.slider("Radius Worst", 7.93, 36.04, 19.38)
    texture_worst = st.slider("Texture Worst", 12.02, 49.54, 25.54)
    perimeter_worst = st.slider("Perimeter Worst", 50.41, 251.2, 122.8)
    area_worst = st.slider("Area Worst", 185.2, 4254.0, 843.7)
    smoothness_worst = st.slider("Smoothness Worst", 0.07117, 0.2226, 0.1559)
    compactness_worst = st.slider("Compactness Worst", 0.02729, 1.058, 0.3459)
    smoothness_mean = st.slider("Smoothness Mean", 0.05263, 0.1634, 0.096)
    compactness_mean = st.slider("Compactness Mean", 0.01938, 0.3454, 0.104)
    concavity_worst = st.slider("Concavity Worst", 0.0, 1.252, 0.3)
    concave_points_worst = st.slider("Concave Points Worst", 0.0, 0.291, 0.1)
    symmetry_worst = st.slider("Symmetry Worst", 0.1565, 0.6638, 0.3)
    fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.05504, 0.2075, 0.1)





    # Get the user input and make a prediction
    data = {'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
        }

    diagnosis = predict_diagnosis(data)

    # Display the prediction to the user
    st.write("The predicted diagnosis is:", diagnosis[0])
    st.write("The features used for prediction are:", diagnosis[1])


# Define function for page 4: Conclusions
def page_conclusions():
    st.title("Conclusions")
    st.write("Here are our conclusions and Recommendations from Project")
    # ...


# Define dictionary of pages and their corresponding functions
pages = {
    "Introduction": page_group_members,
    "Data Set Description": page_data_description,
    "Predictions": page_predictions,
    "Conclusions": page_conclusions
}

# Add navigation menu to sidebar
st.sidebar.title("Navigation")
page_names = list(pages.keys())
selected_page = st.sidebar.selectbox("Go to", page_names)

# Display the selected page with its corresponding function
pages[selected_page]()
