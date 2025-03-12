import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import statsmodels.api as sm

# Set Page Configuration
st.set_page_config(layout="wide")

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
    st.subheader("Filter Data")
    brand = st.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].dropna().unique()))
    gender = st.selectbox("Select Gender", [None] + list(df["Gender"].dropna().unique()))
    income = st.selectbox("Select Income Level", [None] + list(df["Income_Level"].dropna().unique()))
    cluster = st.selectbox("Select Cluster", [None] + list(df["Cluster_Name"].dropna().unique()))

# Apply Filters
filtered_df = df.copy()
if brand:
    filtered_df = filtered_df[filtered_df["Brand_Preference"] == brand]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"] == gender]
if income:
    filtered_df = filtered_df[filtered_df["Income_Level"] == income]
if cluster:
    filtered_df = filtered_df[filtered_df["Cluster_Name"] == cluster]

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
    occasions_counts = filtered_df['Occasions_Of_Buying'].value_counts()
    fig = px.pie(occasions_counts, names=occasions_counts.index, title='Buying Occasions Distribution')
    st.plotly_chart(fig)

elif section == "Basic Attribute Scores":
    st.subheader("Basic Attribute Scores")
    attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']
    avg_scores = filtered_df[attributes].mean()
    fig = px.bar(x=avg_scores.index, y=avg_scores.values, text=avg_scores.values.round(2), title='Basic Attribute Scores')
    st.plotly_chart(fig)
    nps_avg = filtered_df.groupby('Age_Group')['NPS_Score'].mean()
    fig = px.bar(x=nps_avg.index, y=nps_avg.values, text=nps_avg.values.round(2), title='NPS Score by Age Group')
    st.plotly_chart(fig)

elif section == "Regression Analysis":
    st.subheader("Regression Analysis")
    X = filtered_df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
    y = filtered_df['NPS_Score']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    st.text(model.summary())

elif section == "Decision Tree Analysis":
    st.subheader("Decision Tree Analysis")
    X_tree = filtered_df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
    y_tree = filtered_df['NPS_Score'].apply(lambda x: 'Promoter' if x > 8 else 'Detractor/Passive')
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_tree, y_tree)
    fig, ax = plt.subplots(figsize=(12, 6))
    tree.plot_tree(clf, feature_names=X_tree.columns, class_names=clf.classes_, filled=True, ax=ax)
    st.pyplot(fig)

elif section == "Cluster Analysis":
    st.subheader("Customer Segmentation")
    cluster_counts = filtered_df['Cluster_Name'].value_counts(normalize=True) * 100
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, text=cluster_counts.values.round(2), title='Cluster Distribution (%)')
    st.plotly_chart(fig)
    st.write("### Cluster Descriptions")
    for cluster_name in df['Cluster_Name'].unique():
        st.write(f"**{cluster_name}:**", df[df['Cluster_Name'] == cluster_name].describe())

elif section == "View & Download Full Dataset":
    st.subheader("Full Dataset")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="cola_survey_data.csv", mime="text/csv")
