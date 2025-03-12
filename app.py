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

# Set wide layout for better mobile visibility


# Streamlit App Title
st.title("Interactive Cola Consumer Dashboard")

# Cluster Analysis (Precompute and Append to Data)
X_cluster = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)
df['Cluster_Name'] = df['Cluster'].map({0: 'Fizz-Lovers', 1: 'Brand-Conscious Consumers', 2: 'Budget-Friendly Drinkers'})

# Sidebar Filters
with st.sidebar:
    brand = st.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].unique()))
    gender = st.selectbox("Select Gender", [None] + list(df["Gender"].unique()))
    income = st.selectbox("Select Income Level", [None] + list(df["Income_Level"].unique()))
    cluster = st.selectbox("Select Cluster", [None] + list(df["Cluster_Name"].unique()))

# Filter Data
filtered_df = df.copy()
if brand:
    filtered_df = filtered_df[filtered_df["Brand_Preference"] == brand]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"] == gender]
if income:
    filtered_df = filtered_df[filtered_df["Income_Level"] == income]
if cluster:
    filtered_df = filtered_df[filtered_df["Cluster_Name"] == cluster]

# Button States for Toggle Functionality
if 'toggle_state' not in st.session_state:
    st.session_state.toggle_state = {section: False for section in [
        "Demographic Profile", "Brand Metrics", "Basic Attribute Scores", "Regression Analysis", 
        "Decision Tree Analysis", "Cluster Analysis", "View & Download Full Dataset"
    ]}

def toggle_section(section_name):
    st.session_state.toggle_state[section_name] = not st.session_state.toggle_state[section_name]

# Arrange buttons in two columns ensuring proper execution
col1, col2 = st.columns(2)
button_sections = list(st.session_state.toggle_state.keys())

buttons = {}
for index, section in enumerate(button_sections):
    key_name = f"button_{index}"  # Unique key for each button
    with col1 if index % 2 == 0 else col2:
        buttons[section] = st.button(section, key=key_name)

# Ensure proper toggling of sections
for section, button_pressed in buttons.items():
    if button_pressed:
        toggle_section(section)
    if st.session_state.toggle_state[section]:
        st.subheader(section)
        if section == "Demographic Profile":
            age_counts = filtered_df['Age_Group'].value_counts(normalize=True).sort_index() * 100
            fig = px.bar(x=age_counts.index, y=age_counts.values, text=age_counts.values.round(2), title='Age Group Distribution (%)')
            st.plotly_chart(fig)
            fig = px.pie(filtered_df, names='Gender', title='Gender Distribution')
            st.plotly_chart(fig)
            fig = px.pie(filtered_df, names='Income_Level', title='Income Level Distribution')
            st.plotly_chart(fig)
        
        if section == "Brand Metrics":
            brand_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
            fig = px.bar(x=brand_counts.index, y=brand_counts.values.round(2), text=brand_counts.values.round(2), title='Most Often Used Brand')
            st.plotly_chart(fig)
            occasions_counts = filtered_df['Occasions_of_Buying'].value_counts(normalize=True) * 100
            fig = px.bar(x=occasions_counts.index, y=occasions_counts.values.round(2), text=occasions_counts.values.round(2), title='Occasions of Buying')
            st.plotly_chart(fig)
            freq_counts = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
            fig = px.bar(x=freq_counts.index, y=freq_counts.values.round(2), text=freq_counts.values.round(2), title='Frequency of Consumption')
            st.plotly_chart(fig)
        
        if section == "View & Download Full Dataset":
            st.dataframe(filtered_df)
            csv = filtered_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name="cola_survey_data.csv", mime="text/csv")

# Apply and Clear Filters
col3, col4 = st.columns(2)
with col3:
    if st.button("Apply Filter (for Mobile)"):
        brand = st.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].unique()), key='brand_mobile')
        gender = st.selectbox("Select Gender", [None] + list(df["Gender"].unique()), key='gender_mobile')
        income = st.selectbox("Select Income Level", [None] + list(df["Income_Level"].unique()), key='income_mobile')
        cluster = st.selectbox("Select Cluster", [None] + list(df["Cluster_Name"].unique()), key='cluster_mobile')

with col4:
    if st.button("Clear Filters"):
        st.session_state.toggle_state = {section: False for section in st.session_state.toggle_state.keys()}
