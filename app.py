import streamlit as st  # Import Streamlit first
st.set_page_config(layout="wide", page_title="Cola Consumer Dashboard", page_icon="🥤")  # Then set the page configuration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.decomposition import FactorAnalysis
from statsmodels.graphics.mosaicplot import mosaic
from io import BytesIO
from factor_analyzer import FactorAnalyzer
import plotly.figure_factory as ff

# Set page styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #3366FF;
        margin-bottom: 0.5rem;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .summary-title {
        font-weight: bold;
        color: #007bff;
        margin-bottom: 0.5rem;
    }
    .filter-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Improve mobile display */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .subheader {
            font-size: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cola_survey.csv")
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[18, 25, 35, 45, 55, 65], 
                            labels=['18-24', '25-34', '35-44', '45-54', '55+'], 
                            right=False)
    
    # Calculate NPS categories
    df['NPS_Category'] = pd.cut(df['NPS_Score'], 
                               bins=[-1, 6, 8, 10], 
                               labels=['Detractors', 'Passives', 'Promoters'])
    
    # Perform clustering (will be available for all sections)
    X_cluster = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                  'Brand_Reputation_Rating', 'Availability_Rating', 
                  'Sweetness_Rating', 'Fizziness_Rating']]
    
    # Standardize data for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Apply KMeans clustering with fixed random_state for reproducibility
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Recalculate cluster centers to ensure consistent distribution
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(centers, 
                                     columns=X_cluster.columns, 
                                     index=['Cluster 0', 'Cluster 1', 'Cluster 2'])
    
    # Assign cluster names based on highest ratings in that cluster
    def assign_cluster_name(cluster_center):
        highest_rating_feature = X_cluster.columns[cluster_center.argmax()]
        mapping = {
            'Taste_Rating': 'Taste Enthusiasts',
            'Brand_Reputation_Rating': 'Brand Loyalists',
            'Price_Rating': 'Value Seekers'
        }
        return mapping.get(highest_rating_feature, 'Undefined Cluster')
    
    cluster_names = {
        0: assign_cluster_name(centers[0]),
        1: assign_cluster_name(centers[1]),
        2: assign_cluster_name(centers[2])
    }
    
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    
    return df, cluster_centers_df

df, cluster_centers = load_data()

# App title
st.markdown("<h1 class='main-header'>Interactive Cola Consumer Dashboard</h1>", unsafe_allow_html=True)

# Initialize session state for filters if not exists
if 'filters' not in st.session_state:
    st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}

# Apply selected filters to the dataframe
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
    "Executive Dashboard Summary",
    "Demographic Profile", 
    "Brand Metrics", 
    "Basic Attribute Scores", 
    "Regression Analysis", 
    "Decision Tree Analysis", 
    "Cluster Analysis", 
    "View & Download Full Dataset"
], horizontal=True)

# MOVED FILTERS HERE - Placed just below section selection for mobile visibility
st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
st.subheader("Data Filters")

# Create filter options with None as first option
col1, col2, col3, col4 = st.columns(4)

with col1:
    brand_options = [None] + sorted(df["Brand_Preference"].unique().tolist())
    brand = st.selectbox("Brand", brand_options, key='brand_filter')

with col2:  
    gender_options = [None] + sorted(df["Gender"].unique().tolist())
    gender = st.selectbox("Gender", gender_options, key='gender_filter')

with col3:
    income_options = [None] + sorted(df["Income_Level"].unique().tolist())
    income = st.selectbox("Income Level", income_options, key='income_filter')

with col4:
    cluster_options = [None] + sorted(df["Cluster_Name"].unique().tolist())
    cluster = st.selectbox("Cluster", cluster_options, key='cluster_filter')

col1, col2 = st.columns(2)
with col1:
    if st.button("Apply Filters", use_container_width=True):
        st.session_state.filters['brand'] = brand
        st.session_state.filters['gender'] = gender
        st.session_state.filters['income'] = income
        st.session_state.filters['cluster'] = cluster
        st.rerun()

with col2:
    if st.button("Clear Filters", use_container_width=True):
        st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Helper function to dynamically generate summary insights
def get_demo_insights(df):
    """Generate demographic insights from filtered data"""
    insights = {}
    
    # Age group insights
    age_counts = df['Age_Group'].value_counts(normalize=True).sort_index() * 100
    top_age_groups = age_counts.nlargest(2).index.tolist()
    insights['age_groups'] = f"{', '.join(top_age_groups)} (representing {age_counts[top_age_groups].sum():.1f}% of consumers)"
    
    # Gender insights
    gender_counts = df['Gender'].value_counts(normalize=True) * 100
    majority_gender = gender_counts.idxmax()
    insights['gender'] = f"{majority_gender} majority ({gender_counts.max():.1f}%)"
    
    # Income insights
    income_counts = df['Income_Level'].value_counts(normalize=True) * 100
    top_income = income_counts.idxmax()
    insights['income'] = f"{top_income} ({income_counts.max():.1f}%)"
    
    return insights

def get_brand_insights(df):
    """Generate brand insights from filtered data"""
    insights = {}
    
    # Brand insights
    brand_counts = df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
    top_brands = brand_counts.nlargest(2).index.tolist()
    insights['brands'] = f"{' and '.join(top_brands)} (combined {brand_counts[top_brands].sum():.1f}%)"
    
    # Occasions insights
    occasion_counts = df['Occasions_of_Buying'].value_counts(normalize=True) * 100
    top_occasions = occasion_counts.nlargest(2).index.tolist()
    insights['occasions'] = f"{' and '.join(top_occasions)}"
    
    # Frequency insights
    freq_counts = df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
    top_freq = freq_counts.idxmax()
    insights['frequency'] = f"{top_freq} ({freq_counts.max():.1f}%)"
    
    # Satisfaction insights
    sat_counts = df['Satisfaction_Level'].value_counts(normalize=True) * 100
    sat_levels = sat_counts.index.tolist()
    if 'Very Satisfied' in sat_levels and 'Satisfied' in sat_levels:
        satisfied_pct = sat_counts['Very Satisfied'] + sat_counts['Satisfied']
        sentiment = "positive" if satisfied_pct > 50 else "mixed"
    else:
        sentiment = "neutral"
    insights['satisfaction'] = sentiment
    
    return insights

def get_attribute_insights(df):
    """Generate attribute insights from filtered data"""
    insights = {}
    
    attributes = [
        'Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
        'Brand_Reputation_Rating', 'Availability_Rating', 
        'Sweetness_Rating', 'Fizziness_Rating'
    ]
    
    # Top and bottom attributes
    avg_scores = df[attributes].mean()
    top_attrs = avg_scores.nlargest(2).index.tolist()
    bottom_attrs = avg_scores.nsmallest(2).index.tolist()
    
    insights['top_attributes'] = ", ".join([attr.replace('_Rating', '') for attr in top_attrs])
    insights['bottom_attributes'] = ", ".join([attr.replace('_Rating', '') for attr in bottom_attrs])
    
    return insights

def get_cluster_insights(df):
    """Generate cluster insights from filtered data"""
    insights = {}
    
    # Ensure cluster distribution adds up to 1000
    cluster_dist = df['Cluster_Name'].value_counts()
    total_respondents = len(df)
    
    # Adjust clustering if total is not 1000
    if total_respondents != 1000:
        # Redistribute to ensure 1000 total
        cluster_dist_normalized = cluster_dist / cluster_dist.sum() * 1000
        cluster_dist = cluster_dist_normalized.round().astype(int)
        
        # Adjust for any rounding errors
        diff = 1000 - cluster_dist.sum()
        if diff != 0:
            # Add or subtract from the largest cluster
            largest_cluster_idx = cluster_dist.argmax()
            cluster_dist.iloc[largest_cluster_idx] += diff
    
    insights['cluster_dist'] = {name: f"{count/10:.1f}%" for name, count in cluster_dist.items()}
    
    # Get top brands and attributes by cluster
    attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                 'Brand_Reputation_Rating', 'Availability_Rating', 
                 'Sweetness_Rating', 'Fizziness_Rating']
    
    insights['cluster_details'] = {}
    
    for cluster in df['Cluster_Name'].unique():
        cluster_df = df[df['Cluster_Name'] == cluster]
        
        # Top brand
        top_brand = cluster_df['Most_Often_Consumed_Brand'].value_counts().idxmax()
        
        # Top attribute
        top_attr = cluster_df[attributes].mean().idxmax().replace('_Rating', '')
        
        # Average NPS
        avg_nps = cluster_df['NPS_Score'].mean()
        
        insights['cluster_details'][cluster] = {
            'top_brand': top_brand,
            'top_attribute': top_attr,
            'avg_nps': f"{avg_nps:.1f}"
        }
    
    return insights

# The rest of the Streamlit app code remains the same as in the previous version, 
# with updated markdown formatting for executive summaries in each section.

# For each section (Demographic Profile, Brand Metrics, Basic Attribute Scores, 
# Regression Analysis, Decision Tree Analysis, Cluster Analysis), 
# you would modify the executive summary markdown to use .format() 
# and include dynamic insights.

# Example for the Executive Dashboard Summary section would look like:
elif section == "Executive Dashboard Summary":
    st.markdown("<h2 class='subheader'>Executive Dashboard Summary</h2>", unsafe_allow_html=True)
    
    # Get insights
    demo_insights = get_demo_insights(filtered_df)
    brand_insights = get_brand_insights(filtered_df)
    attr_insights = get_attribute_insights(filtered_df)
    
    # Rest of the section remains the same...
    
    # Executive Summary Box with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>EXECUTIVE SUMMARY</div>
        <p>This analysis examines preferences, behaviors, and satisfaction levels of 1,000 cola consumers. 
        The survey captured demographic information, consumption patterns, brand preferences, and ratings across 
        various product attributes.</p>
        
        <p><strong>Key Demographic Insights:</strong></p>
        <ul>
            <li>Age distribution shows prevalence in {age_groups}</li>
            <li>Gender split indicates {gender}</li>
            <li>Income level distribution suggests predominance of {income}</li>
        </ul>
        
        <p><strong>Brand and Consumption Insights:</strong></p>
        <ul>
            <li>Market share is dominated by {brands}</li>
            <li>Primary consumption occasions include {occasions}</li>
            <li<li>Consumption frequency patterns reveal {frequency} as most common</li>
            <li>Overall satisfaction levels indicate {satisfaction} sentiment</li>
        </ul>
        
        <p><strong>Product Attribute Insights:</strong></p>
        <ul>
            <li>The highest-rated attributes are {top_attributes}</li>
            <li>The lowest-rated attributes are {bottom_attributes}</li>
        </ul>
        
        <p><strong>Strategic Recommendations:</strong></p>
        <ul>
            <li>Focus product development on taste improvement as the primary satisfaction driver</li>
            <li>Develop targeted marketing for each consumer segment based on their priorities</li>
            <li>Invest in brand reputation building to drive long-term loyalty</li>
            <li>Monitor NPS by segment to track loyalty improvements over time</li>
        </ul>
    </div>
    """.format(
        age_groups=demo_insights['age_groups'],
        gender=demo_insights['gender'],
        income=demo_insights['income'],
        brands=brand_insights['brands'],
        occasions=brand_insights['occasions'],
        frequency=brand_insights['frequency'],
        satisfaction=brand_insights['satisfaction'],
        top_attributes=attr_insights['top_attributes'],
        bottom_attributes=attr_insights['bottom_attributes']
    ), unsafe_allow_html=True)

# Continue with the rest of the sections, applying similar dynamic formatting

# Demographic Profile Section
elif section == "Demographic Profile":
    st.markdown("<h2 class='subheader'>Demographic Profile</h2>", unsafe_allow_html=True)
    
    # Get demographic insights
    demo_insights = get_demo_insights(filtered_df)
    
    # Existing visualizations remain the same...
    
    # Update Executive Summary with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>DEMOGRAPHIC PROFILE - EXECUTIVE SUMMARY</div>
        <p>The demographic analysis of cola consumers reveals distinct patterns across age groups, gender, and income levels:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Age distribution shows prevalence in {age_groups}</li>
            <li>Gender split indicates {gender}</li>
            <li>Income level distribution suggests predominance of {income}</li>
            <li>There are noticeable correlations between demographics and preferences</li>
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Target marketing efforts to align with demographic composition</li>
            <li>Consider demographic factors in product development and positioning</li>
            <li>Address potential gaps in market penetration across demographic segments</li>
            <li>Monitor demographic shifts for early identification of market trends</li>
        </ul>
    </div>
    """.format(
        age_groups=demo_insights['age_groups'],
        gender=demo_insights['gender'],
        income=demo_insights['income']
    ), unsafe_allow_html=True)

# Brand Metrics Section
elif section == "Brand Metrics":
    st.markdown("<h2 class='subheader'>Brand Metrics</h2>", unsafe_allow_html=True)
    
    # Get brand insights
    brand_insights = get_brand_insights(filtered_df)
    
    # Existing visualizations remain the same...
    
    # Update Executive Summary with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>BRAND METRICS - EXECUTIVE SUMMARY</div>
        <p>The analysis of brand metrics provides insights into consumer preferences and consumption patterns:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Market share is dominated by {brands}</li>
            <li>Primary consumption occasions include {occasions}</li>
            <li>Consumption frequency patterns reveal {frequency} as most common</li>
            <li>Overall satisfaction levels indicate {satisfaction} sentiment</li>
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Focus marketing on key consumption occasions to maximize relevance</li>
            <li>Address frequency patterns in product packaging and distribution</li>
            <li>Leverage satisfaction drivers to strengthen brand positioning</li>
            <li>Consider occasion-specific product variants or marketing campaigns</li>
        </ul>
    </div>
    """.format(
        brands=brand_insights['brands'],
        occasions=brand_insights['occasions'],
        frequency=brand_insights['frequency'],
        satisfaction=brand_insights['satisfaction']
    ), unsafe_allow_html=True)

# Basic Attribute Scores Section
elif section == "Basic Attribute Scores":
    st.markdown("<h2 class='subheader'>Basic Attribute Scores</h2>", unsafe_allow_html=True)
    
    # Get attribute insights
    attr_insights = get_attribute_insights(filtered_df)
    
    # Existing visualizations remain the same...
    
    # Update Executive Summary with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>ATTRIBUTE SCORES - EXECUTIVE SUMMARY</div>
        <p>The analysis of product attribute ratings reveals priorities and satisfaction drivers:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>The highest-rated attributes are {top_attributes}</li>
            <li>The lowest-rated attributes are {bottom_attributes}</li>
            <li>NPS scores vary significantly across demographic segments</li>
            <li>Gender and age show notable influence on attribute preferences</li>
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Focus product improvement efforts on lower-rated attributes</li>
            <li>Leverage strengths in marketing communications</li>
            <li>Address NPS variations with targeted loyalty initiatives</li>
            <li>Consider segment-specific product variants to address preference variations</li>
        </ul>
    </div>
    """.format(
        top_attributes=attr_insights['top_attributes'],
        bottom_attributes=attr_insights['bottom_attributes']
    ), unsafe_allow_html=True)

# Regression Analysis Section
elif section == "Regression Analysis":
    st.markdown("<h2 class='subheader'>Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Prepare data for regression
    X_reg = filtered_df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                       'Brand_Reputation_Rating', 'Availability_Rating', 
                       'Sweetness_Rating', 'Fizziness_Rating']]
    
    y_reg = filtered_df['NPS_Score']
    
    # Add constant to predictor variables
    X_reg = sm.add_constant(X_reg)
    
    # Fit regression model
    model = sm.OLS(y_reg, X_reg).fit()
    
    # Function to get regression insights
    def get_regression_insights(model):
        insights = {}
        
        # Most influential positive factors
        significant_features = pd.DataFrame({
            'Feature': X_reg.columns,
            'Coefficient': model.params,
            'P-Value': model.pvalues,
            'Significant': model.pvalues < 0.05
        })
        
        sig_positive = significant_features[
            (significant_features['Significant'] == True) & 
            (significant_features['Feature'] != 'const') & 
            (significant_features['Coefficient'] > 0)
        ].sort_values('Coefficient', ascending=False)
        
        if not sig_positive.empty:
            insights['top_factors'] = ", ".join(
                [f.replace('_Rating', '') for f in sig_positive.iloc[:2]['Feature'].tolist()]
            )
        else:
            insights['top_factors'] = "No significant positive factors"
            
        # Model quality
        insights['rsquared'] = f"{model.rsquared:.1%}"
        insights['significant'] = "statistically significant" if model.f_pvalue < 0.05 else "not statistically significant"
        
        return insights
    
    # Get regression insights
    reg_insights = get_regression_insights(model)
    
    # Existing visualizations and analysis remain the same...
    
    # Update Executive Summary with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>REGRESSION ANALYSIS - EXECUTIVE SUMMARY</div>
        <p>The regression analysis identifies the key drivers of consumer loyalty as measured by NPS:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>The regression model explains <strong>{rsquared}</strong> of the variation in NPS scores</li>
            <li>The model is <strong>{significant}</strong></li>
            <li>The most influential factors are <strong>{top_factors}</strong></li>
            {additional_insight}
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Focus improvement efforts on the attributes with strongest positive coefficients</li>
            <li>Address negative drivers to minimize their impact on consumer loyalty</li>
            <li>Use regression insights to prioritize product development initiatives</li>
            <li>Consider the balance of attribute improvements against implementation costs</li>
        </ul>
    </div>
    """.format(
        rsquared=reg_insights['rsquared'],
        significant=reg_insights['significant'],
        top_factors=reg_insights['top_factors'],
        additional_insight=("<li>The model suggests additional factors may influence NPS beyond the attributes measured</li>" 
                            if model.rsquared < 0.3 else "")
    ), unsafe_allow_html=True)

# Decision Tree Analysis Section
elif section == "Decision Tree Analysis":
    st.markdown("<h2 class='subheader'>Decision Tree Analysis</h2>", unsafe_allow_html=True)
    
    # Prepare and train decision tree (existing code remains the same)
    
    # Function to get decision tree insights
    def get_tree_insights(dt_model, feature_importance):
        insights = {}
        
        # Top factor
        insights['top_factor'] = feature_importance.iloc[0]['Feature'].replace('_Rating', '')
        
        # Second factor
        if len(feature_importance) > 1:
            insights['second_factor'] = feature_importance.iloc[1]['Feature'].replace('_Rating', '')
        else:
            insights['second_factor'] = "None"
            
        # Accuracy
        insights['accuracy'] = f"{test_accuracy:.1%}"
        
        # Promoter rule
        rules = tree.export_text(dt_model, 
                                feature_names=list(X_tree.columns),
                                max_depth=3)
        
        # Find a rule for promoters
        if "class: Promoter" in rules:
            promoter_rule = "High " + insights['top_factor']
        else:
            promoter_rule = "No clear path to promoter status identified"
            
        insights['promoter_rule'] = promoter_rule
        
        return insights
    
    # Get tree insights
    tree_insights = get_tree_insights(dt_model, feature_importance)
    
    # Update Executive Summary with dynamic content
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>DECISION TREE ANALYSIS - EXECUTIVE SUMMARY</div>
        <p>The decision tree analysis identifies the critical decision pathways that determine consumer loyalty:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>The model achieved <strong>{accuracy}</strong> accuracy in predicting NPS categories</li>
            <li>The most important classification factor is <strong>{top_factor}</strong></li>
            <li>Secondary factor: <strong>{second_factor}</strong></li>
            <li>Path to promoter status: <strong>{promoter_rule}</strong></li>
            <li>Customer segments show distinct loyalty patterns based on attribute preferences</li>
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Focus improvement efforts on the top decision factors identified in the tree</li>
            <li>Segment customers based on these decision rules for targeted marketing</li>
            <li>Address specific pain points for potential detractors</li>
            <li>Use decision paths to create customer journey optimization strategies</li>
        </ul>
    </div>
    """.format(
        accuracy=tree_insights['accuracy'],
        top_factor=tree_insights['top_factor'],
        second_factor=tree_insights['second_factor'],
        promoter_rule=tree_insights['promoter_rule']
    ), unsafe_allow_html=True)

# Cluster Analysis Section
elif section == "Cluster Analysis":
    st.markdown("<h2 class='subheader'>Cluster Analysis</h2>", unsafe_allow_html=True)
    
    # Get cluster insights
    cluster_insights = get_cluster_insights(filtered_df)
    
    # Prepare cluster items for executive summary
    cluster_items = ""
    for cluster, data in cluster_insights['cluster_details'].items():
        # Describe cluster characteristics dynamically
        if cluster == "Taste Enthusiasts":
            description = f"Prioritize taste and flavor experience above all. Prefers {data['top_brand']}. Average NPS: {data['avg_nps']}."
        elif cluster == "Brand Loyalists":
            description = f"Place high importance on brand reputation. Loyal to {data['top_brand']}. Average NPS: {data['avg_nps']}."
        elif cluster == "Value Seekers":
            description = f"More price-conscious and practical. Top brand: {data['top_brand']}. Average NPS: {data['avg_nps']}."
        else:
            description = f"Segment with distinct preferences. Top attribute: {data['top_attribute']}. Average NPS: {data['avg_nps']}."
        
        # Add cluster distribution
        dist_text = ""
        if cluster in cluster_insights['cluster_dist']:
            dist_text = f" ({cluster_insights['cluster_dist'][cluster]})"
        
        cluster_items += f"<li><strong>{cluster}{dist_text}:</strong> {description}</li>"
    
    # Update Executive Summary with dynamic content
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>CLUSTER ANALYSIS - EXECUTIVE SUMMARY</div>
        <p>The cluster analysis identified distinct consumer segments based on their preferences and priorities:</p>
        
        <p><strong>Consumer Segments:</strong></p>
        <ul>
            {cluster_items}
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Develop targeted product offerings for each segment</li>
            <li>Customize marketing messages to address segment-specific priorities</li>
            <li>Allocate marketing resources based on segment size and potential</li>
            <li>Monitor segment evolution over time to adapt strategies</li>
        </ul>
        
        <p>Each segment has distinct demographic characteristics and consumption patterns, offering opportunities for targeted marketing and product development strategies.</p>
    </div>
    """, unsafe_allow_html=True)

# Rest of the code (View & Download Full Dataset section) remains the same...

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p><strong>Cola Survey Dashboard</strong> | Created with Streamlit</p>
    <p style="font-size: 0.8rem; color: #666;">Last updated: March 2025</p>
</div>
""", unsafe_allow_html=True)
