# import streamlit as st
# import pandas as pd







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
def page_predictions():
    st.title("Predictions")
    st.write("Here's where you can make predictions based on the data:")

    # Load the saved model from disk
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)

    # Define a function to make predictions on new data
    def predict_diagnosis(data):
        X_new = pd.DataFrame(data, index=[0])
        preprocessor = load_preprocessor() # load preprocessor from pickle file
        X_new = preprocessor.transform(X_new)
        feature_names = preprocessor.get_feature_names_out() # call get_feature_names_out directly on preprocessor
        svm_model = load_svm_model() # load SVM model from pickle file
        y_pred = svm_model.predict(X_new)
        return y_pred[0], feature_names


    # Load data
    df = pd.read_csv('data.csv')



    st.write("Enter the values below to predict the diagnosis.")
    radius_mean = st.slider("Radius Mean", 6.98, 28.11, 16.17)
    texture_mean = st.slider("Texture Mean", 9.71, 39.28, 21.41)
    perimeter_mean = st.slider("Perimeter Mean", 43.79, 188.5, 91.97)
    area_mean = st.slider("Area Mean", 143.5, 2501.0, 654.89)
    concavity_mean = st.slider("Concavity Mean", 0.0, 0.4275, 0.1)
    concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.2012, 0.1)

    # Get the user input and make a prediction
    data = {'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'concavity_mean': concavity_mean,
            'concave points_mean': concave_points_mean}
    diagnosis = predict_diagnosis(data)

    # Display the prediction to the user
    st.write("The predicted diagnosis is:", diagnosis)


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
