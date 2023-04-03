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
    import streamlit as st
    import pandas as pd
    #rom sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Load the Wisconsin Breast Cancer dataset
    data = pd.read_csv('breast_cancer.csv')


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
