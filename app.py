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
    st.title("PREDICTIVE MODEL FOR BREAST CANCER DIAGNOSIS USING MACHINE LEARNING TECHNIQUES")
    st.title("Group 2's Final Year Project's Data Analysis/Machine Learning Component\n\nComputer Science Department, School of Computing and Engineering Sciences, Babcock University, Ilishan-Remo, Ogun State, Nigeria.\n\nGroup 2 Members:\n\n1. Ogunsusi, Adetunji Gabriel 19/2500\n2. Okoye, Adaora Jessica 19/0686\n\nGroup Project Supervisor: Ernest E. Onuiri, PHD\n\r")
    # Load image from URL
    url = "https://images.pexels.com/photos/579474/pexels-photo-579474.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    st.image(url, caption=" ")
    
# Define function for page 1: Data Set Description
def page_data_description():
    st.title("Data Set Description")
    st.write("Here is a description of our data set:")
    
    # Load data set
    df = pd.read_csv('data.csv')
    
    # Display data set summary and description
    st.subheader("Data Set Summary")
    st.write(df.head(4))

    st.subheader("\n\n\nData Set Description")
    st.write(df.describe())
    st.write("This dataset contains information about the Wisconsin Breast Cancer, which was collected from breast cancer patients at the University of Wisconsin Hospitals, Madison, in the USA. The dataset consists of measurements of 30 different features, including the radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension of the cell nuclei present in images of the breast tissue. Each feature is represented as a real-valued number. The dataset also includes a binary target variable indicating whether the tumor is benign or malignant. There are a total of 569 instances in the dataset, with 357 instances labeled as benign and 212 instances labeled as malignant. The dataset is commonly used in machine learning research to train models for breast cancer diagnosis and prediction.")
    feature_descriptions = {
    'radius': 'The mean distance from the center of the nucleus to the perimeter of the cell.',
    'texture': 'The standard deviation of gray-scale values in the image, which is a measure of the variation in the intensity of the pixels.',
    'perimeter': 'The total length of the cell perimeter.',
    'area': 'The total area of the cell.',
    'smoothness': 'A measure of the local variation in radius lengths.',
    'compactness': 'A measure of the perimeter squared divided by the area - a high value indicates that the cells are compact.',
    'concavity': 'The severity of concave portions of the contour of the cell.',
    'concave points': 'The number of concave portions of the contour of the cell.',
    'symmetry': 'A measure of the symmetry of the cell.',
    'fractal dimension': 'A measure of the complexity of the cell boundary.',
    'radius_mean': 'The mean value of the radius feature for the 3 largest cell nuclei in the image.',
    'texture_mean': 'The mean value of the texture feature for the 3 largest cell nuclei in the image.',
    'perimeter_mean': 'The mean value of the perimeter feature for the 3 largest cell nuclei in the image.',
    'area_mean': 'The mean value of the area feature for the 3 largest cell nuclei in the image.',
    'smoothness_mean': 'The mean value of the smoothness feature for the 3 largest cell nuclei in the image.',
    'compactness_mean': 'The mean value of the compactness feature for the 3 largest cell nuclei in the image.',
    'concavity_mean': 'The mean value of the concavity feature for the 3 largest cell nuclei in the image.',
    'concave points_mean': 'The mean value of the concave points feature for the 3 largest cell nuclei in the image.',
    'symmetry_mean': 'The mean value of the symmetry feature for the 3 largest cell nuclei in the image.',
    'fractal_dimension_mean': 'The mean value of the fractal dimension feature for the 3 largest cell nuclei in the image.',
    'radius_se': 'The standard error of the radius mean feature calculated across all the cells in the image.',
    'texture_se': 'The standard error of the texture mean feature calculated across all the cells in the image.',
    'perimeter_se': 'The standard error of the perimeter mean feature calculated across all the cells in the image.',
    'area_se': 'The standard error of the area mean feature calculated across all the cells in the image.',
    'smoothness_se': 'The standard error of the smoothness mean feature calculated across all the cells in the image.',
    'compactness_se': 'The standard error of the compactness mean feature calculated across all the cells in the image.',
    'concavity_se': 'The standard error of the concavity mean feature calculated across all the cells in the image.',
    'concave points_se': 'The standard error of the concave points mean feature calculated across all the cells in the image.',
    'symmetry_se': 'The standard error of the symmetry mean feature calculated across all the cells in the image.',
    'fractal_dimension_se': 'The standard error of the fractal dimension mean feature calculated across all the cells in the image.'}
    
    st.write("Feature Descriptions:")
    for feature, description in feature_descriptions.items():
        st.markdown(f"- **{feature}:** {description}")








    
    # Display data cleaning process and visualization
    st.subheader("Data Cleaning")
    st.write("Here's how we cleaned the data:")

    
    # Drop columns with missing values
    df = df.dropna(axis=1)
    st.write("We dropped the columns with missing values.")
    
    # Label encode the diagnosis column
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    st.write("We label encoded the diagnosis column: 1 for malignant 0 for Benign")
    
    # Display cleaned data set
    st.subheader("Cleaned Data Set")
    st.write(df.head(5))

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
    st.title('Breast Cancer Prediction App')

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
    st.write("In this project, we used machine learning to develop a breast cancer prediction model using the Wisconsin Breast Cancer dataset. We tested four different models, including the CART, Gaussian NB, KNN, and SVM, and found that the SVM model performed the best with an accuracy score of 98.2%. We then deployed the model using Streamlit, which creates an interactive data science application that allows users to input patient data and receive a prediction of their breast cancer risk. While the deployed model has the potential to be used as a tool for early detection and treatment of breast cancer, there are several limitations to be considered, such as those related to data quality, generalization, and the complexity of the disease. Future work could involve improving the model's performance by incorporating more diverse datasets, additional features, and exploring more advanced machine learning techniques. Overall, this project demonstrates the potential of machine learning in the field of healthcare, and highlights the importance of continued research in the area of breast cancer prediction and treatment.")


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
