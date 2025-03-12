import streamlit as st  # Import Streamlit first
st.set_page_config(layout="wide")  # Then set the page configuration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import statsmodels.api as sm

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cola_survey.csv")
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-24', '25-34', '35-44', '45-54', '55-64'], ordered=True)
    return df

df = load_data()

# Streamlit App Title
st.title("Interactive Cola Consumer Dashboard")

# Cluster Analysis (Precompute and Append to Data)
X_cluster = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)
df['Cluster_Name'] = df['Cluster'].map({0: 'Fizz-Lovers', 1: 'Brand-Conscious Consumers', 2: 'Budget-Friendly Drinkers'})

# Sidebar Filters
with st.sidebar:
    brand = st.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].unique()), key='brand_sidebar')
    gender = st.selectbox("Select Gender", [None] + list(df["Gender"].unique()), key='gender_sidebar')
    income = st.selectbox("Select Income Level", [None] + list(df["Income_Level"].unique()), key='income_sidebar')
    cluster = st.selectbox("Select Cluster", [None] + list(df["Cluster_Name"].unique()), key='cluster_sidebar')

# Store filter selections in session state
if 'filters' not in st.session_state:
    st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}

# Apply selected filters
filtered_df = df.copy()
if st.session_state.filters['brand']:
    filtered_df = filtered_df[filtered_df["Brand_Preference"] == st.session_state.filters['brand']]
if st.session_state.filters['gender']:
    filtered_df = filtered_df[filtered_df["Gender"] == st.session_state.filters['gender']]
if st.session_state.filters['income']:
    filtered_df = filtered_df[filtered_df["Income_Level"] == st.session_state.filters['income']]
if st.session_state.filters['cluster']:
    filtered_df = filtered_df[filtered_df["Cluster_Name"] == st.session_state.filters['cluster']]

# Section Selection using Radio Buttons
section = st.radio("Select Analysis Section", [
    "Demographic Profile", "Brand Metrics", "Basic Attribute Scores", "Regression Analysis", 
    "Decision Tree Analysis", "Cluster Analysis", "View & Download Full Dataset"
])

# Display Selected Section
if section == "Demographic Profile":
    st.subheader("Demographic Profile")
    age_counts = filtered_df['Age_Group'].value_counts(normalize=True).sort_index() * 100
    fig = px.bar(x=age_counts.index, y=age_counts.values, text=age_counts.values.round(2), title='Age Group Distribution (%)')
    st.plotly_chart(fig)
    fig = px.pie(filtered_df, names='Gender', title='Gender Distribution')
    st.plotly_chart(fig)
    fig = px.pie(filtered_df, names='Income_Level', title='Income Level Distribution')
    st.plotly_chart(fig)

elif section == "Brand Metrics":
    st.subheader("Brand Metrics")
    brand_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
    fig = px.bar(x=brand_counts.index, y=brand_counts.values.round(2), text=brand_counts.values.round(2), title='Most Often Used Brand')
    st.plotly_chart(fig)

elif section == "Basic Attribute Scores":
    st.subheader("Basic Attribute Scores")
    attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']
    avg_scores = filtered_df[attributes].mean()
    fig = px.bar(x=avg_scores.index, y=avg_scores.values, text=avg_scores.values.round(2), title='Basic Attribute Scores')
    st.plotly_chart(fig)

elif section == "View & Download Full Dataset":
    st.subheader("Full Dataset")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="cola_survey_data.csv", mime="text/csv")

# Apply and Clear Filters
col1, col2 = st.columns(2)
with col1:
    if st.button("Apply Filter (for Mobile)"):
        st.session_state.filters['brand'] = brand
        st.session_state.filters['gender'] = gender
        st.session_state.filters['income'] = income
        st.session_state.filters['cluster'] = cluster
        st.rerun()

with col2:
    if st.button("Clear Filters"):
        st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}
        st.rerun()
