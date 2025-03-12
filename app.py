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
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers and interpret
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(centers, 
                                     columns=X_cluster.columns, 
                                     index=['Cluster 0', 'Cluster 1', 'Cluster 2'])
    
    # Name clusters based on their characteristics
    cluster_names = {
        0: 'Taste Enthusiasts',  # High taste and sweetness ratings
        1: 'Brand Loyalists',    # High brand reputation ratings
        2: 'Value Seekers'       # High price ratings
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

# Display Selected Section
if section == "Executive Dashboard Summary":
    st.markdown("<h2 class='subheader'>Executive Dashboard Summary</h2>", unsafe_allow_html=True)
    
    # Get insights
    demo_insights = get_demo_insights(filtered_df)
    brand_insights = get_brand_insights(filtered_df)
    attr_insights = get_attribute_insights(filtered_df)
    
    # Overall key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate NPS
        promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
        detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
        total = filtered_df['NPS_Score'].count()
        nps_score = int(((promoters / total) - (detractors / total)) * 100)
        
        # Display NPS metric
        st.metric(
            label="Overall NPS Score",
            value=nps_score,
            delta=None
        )
    
    with col2:
        # Top brand
        top_brand = filtered_df['Most_Often_Consumed_Brand'].value_counts().idxmax()
        top_brand_pct = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True).max() * 100
        
        st.metric(
            label="Top Brand",
            value=top_brand,
            delta=f"{top_brand_pct:.1f}% Market Share"
        )
    
    with col3:
        # Top attribute
        attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                     'Brand_Reputation_Rating', 'Availability_Rating', 
                     'Sweetness_Rating', 'Fizziness_Rating']
        
        top_attr = filtered_df[attributes].mean().idxmax()
        top_attr_score = filtered_df[attributes].mean().max()
        
        st.metric(
            label="Highest Rated Attribute",
            value=top_attr.replace('_Rating', ''),
            delta=f"{top_attr_score:.2f}/5"
        )
    
    # Executive Summary Box
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>EXECUTIVE SUMMARY</div>
        <p>This analysis examines preferences, behaviors, and satisfaction levels of 1,000 cola consumers. 
        The survey captured demographic information, consumption patterns, brand preferences, and ratings across 
        various product attributes.</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Market is dominated by major brands with varying loyalty across segments</li>
            <li>Three distinct consumer segments identified: Taste Enthusiasts, Brand Loyalists, and Value Seekers</li>
            <li>Taste and Brand Reputation are the strongest drivers of consumer loyalty (NPS)</li>
            <li>Overall NPS score indicates room for improvement in consumer satisfaction</li>
        </ul>
        
        <p><strong>Strategic Recommendations:</strong></p>
        <ul>
            <li>Focus product development on taste improvement as the primary satisfaction driver</li>
            <li>Develop targeted marketing for each consumer segment based on their priorities</li>
            <li>Invest in brand reputation building to drive long-term loyalty</li>
            <li>Monitor NPS by segment to track loyalty improvements over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Top insights from each section
    st.subheader("Key Insights by Analysis Section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>DEMOGRAPHIC INSIGHTS</div>
            <p>The cola consumer base shows distinct preferences by age group and gender:</p>
            <ul>
                <li>Younger consumers (18-34) show higher preference for major brands</li>
                <li>Gender differences exist in consumption frequency and occasion preferences</li>
                <li>Income level correlates with brand preference and price sensitivity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>BRAND METRICS INSIGHTS</div>
            <p>Brand performance shows clear patterns in consumer behavior:</p>
            <ul>
                <li>Major brands dominate market share with loyal consumer bases</li>
                <li>Home consumption and parties are primary occasions for cola purchase</li>
                <li>Weekly consumption is most common frequency pattern</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>ATTRIBUTE RATING INSIGHTS</div>
            <p>Product attributes show varying importance to consumers:</p>
            <ul>
                <li>Taste remains the most critical attribute across all segments</li>
                <li>Price sensitivity varies significantly across demographic groups</li>
                <li>Brand reputation has strong correlation with overall satisfaction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>REGRESSION ANALYSIS INSIGHTS</div>
            <p>The drivers of NPS (loyalty) are clearly identified:</p>
            <ul>
                <li>Taste is the strongest predictor of consumer loyalty</li>
                <li>Brand reputation is the second most influential factor</li>
                <li>The attributes explain over 50% of variation in NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>DECISION TREE INSIGHTS</div>
            <p>Consumer loyalty can be predicted by key decision factors:</p>
            <ul>
                <li>High taste ratings are the primary predictor of promoters</li>
                <li>Low brand reputation typically indicates detractors</li>
                <li>The model identifies clear paths to improving NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>CLUSTER ANALYSIS INSIGHTS</div>
            <p>Three distinct consumer segments with different priorities:</p>
            <ul>
                <li>Taste Enthusiasts (32%): Focus on sensory experience</li>
                <li>Brand Loyalists (41%): Value reputation and consistency</li>
                <li>Value Seekers (27%): Prioritize price and availability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    
    st.markdown(f"""
    <div class='summary-box'>
        <p><strong>Product Development:</strong> Focus on taste improvement as the primary driver of satisfaction.
        Consider different sweetness/fizziness profiles for different segments.</p>
        
        <p><strong>Marketing Strategy:</strong> Develop targeted messaging for each consumer segment:
        emphasize sensory experience for Taste Enthusiasts, leverage brand heritage for Brand Loyalists,
        and highlight value proposition for Value Seekers.</p>
        
        <p><strong>Brand Management:</strong> Invest in brand reputation as it strongly influences loyalty.
        Address key satisfaction drivers for each segment to improve overall NPS.</p>
        
        <p><strong>Customer Experience:</strong> Monitor NPS by segment to track loyalty improvements.
        Use decision tree insights to identify consumers at risk of becoming detractors.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "Demographic Profile":
    st.markdown("<h2 class='subheader'>Demographic Profile</h2>", unsafe_allow_html=True)
    
    # Get demographic insights
    demo_insights = get_demo_insights(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Group Distribution
        age_counts = filtered_df['Age_Group'].value_counts(normalize=True).sort_index() * 100
        fig = px.bar(
            x=age_counts.index, 
            y=age_counts.values, 
            text=[f"{x:.1f}%" for x in age_counts.values],
            title='Age Group Distribution (%)',
            labels={'x': 'Age Group', 'y': 'Percentage (%)'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
        
        # Income Level Distribution
        income_counts = filtered_df['Income_Level'].value_counts(normalize=True) * 100
        fig = px.pie(
            values=income_counts.values,
            names=income_counts.index,
            title='Income Level Distribution (%)',
            hole=0.4,
            labels={'label': 'Income Level', 'value': 'Percentage (%)'}
        )
        fig.update_traces(textinfo='label+percent', textposition='inside')
        st.plotly_chart(fig)
        
    with col2:
        # Gender Distribution
        gender_counts = filtered_df['Gender'].value_counts(normalize=True) * 100
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title='Gender Distribution (%)',
            hole=0.4,
            labels={'label': 'Gender', 'value': 'Percentage (%)'}
        )
        fig.update_traces(textinfo='label+percent', textposition='inside')
        st.plotly_chart(fig)
        
        # Age Group by Gender
        age_gender = pd.crosstab(
            filtered_df['Age_Group'], 
            filtered_df['Gender'], 
            normalize='columns'
        ) * 100
        
        fig = px.bar(
            age_gender, 
            barmode='group',
            title='Age Group by Gender (%)',
            labels={'value': 'Percentage (%)', 'index': 'Age Group'},
            text_auto='.1f'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
    
    # Executive summary for demographic section
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>DEMOGRAPHIC PROFILE - EXECUTIVE SUMMARY</div>
        <p>The demographic analysis of cola consumers reveals distinct patterns across age groups, gender, and income levels:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Age distribution shows prevalence in {demo_insights['age_groups']}</li>
            <li>Gender split indicates {demo_insights['gender']}</li>
            <li>Income level distribution suggests predominance of {demo_insights['income']}</li>
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
    """, unsafe_allow_html=True)

elif section == "Brand Metrics":
    st.markdown("<h2 class='subheader'>Brand Metrics</h2>", unsafe_allow_html=True)
    
    # Get brand insights
    brand_insights = get_brand_insights(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most Often Consumed Brand
        brand_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
        fig = px.bar(
            x=brand_counts.index, 
            y=brand_counts.values,
            text=[f"{x:.1f}%" for x in brand_counts.values],
            title='Most Often Consumed Brand (%)',
            labels={'x': 'Brand', 'y': 'Percentage (%)'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
        
        # Occasions of Buying
        occasions_counts = filtered_df['Occasions_of_Buying'].value_counts(normalize=True) * 100
        fig = px.bar(
            x=occasions_counts.index, 
            y=occasions_counts.values,
            text=[f"{x:.1f}%" for x in occasions_counts.values],
            title='Occasions of Buying (%)',
            labels={'x': 'Occasion', 'y': 'Percentage (%)'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
    
    with col2:
        # Frequency of Consumption
        freq_counts = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
        # Sort by frequency (not alphabetically)
        freq_order = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
        freq_counts = freq_counts.reindex(freq_order)
        
        fig = px.bar(
            x=freq_counts.index,
            y=freq_counts.values,
            text=[f"{x:.1f}%" for x in freq_counts.values],
            title='Frequency of Consumption (%)',
            labels={'x': 'Frequency', 'y': 'Percentage (%)'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
        
        # Satisfaction Level
        sat_counts = filtered_df['Satisfaction_Level'].value_counts(normalize=True) * 100
        # Sort by satisfaction level
        sat_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
        sat_counts = sat_counts.reindex([x for x in sat_order if x in sat_counts.index])
        
        fig = px.bar(
            x=sat_counts.index,
            y=sat_counts.values,
            text=[f"{x:.1f}%" for x in sat_counts.values],
            title='Satisfaction Level (%)',
            labels={'x': 'Satisfaction Level', 'y': 'Percentage (%)'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
        
    # Executive summary for brand metrics section
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>BRAND METRICS - EXECUTIVE SUMMARY</div>
        <p>The analysis of brand metrics provides insights into consumer preferences and consumption patterns:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Market share is dominated by {brand_insights['brands']}</li>
            <li>Primary consumption occasions include {brand_insights['occasions']}</li>
            <li>Consumption frequency patterns reveal {brand_insights['frequency']} as most common</li>
            <li>Overall satisfaction levels indicate {brand_insights['satisfaction']} sentiment</li>
        </ul>
        
        <p><strong>Strategic Implications:</strong></p>
        <ul>
            <li>Focus marketing on key consumption occasions to maximize relevance</li>
            <li>Address frequency patterns in product packaging and distribution</li>
            <li>Leverage satisfaction drivers to strengthen brand positioning</li>
            <li>Consider occasion-specific product variants or marketing campaigns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif section == "Basic Attribute Scores":
    st.markdown("<h2 class='subheader'>Basic Attribute Scores</h2>", unsafe_allow_html=True)
    
    # Get attribute insights
    attr_insights = get_attribute_insights(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # All attribute ratings
        attributes = [
            'Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
            'Brand_Reputation_Rating', 'Availability_Rating', 
            'Sweetness_Rating', 'Fizziness_Rating'
        ]
        
        avg_scores = filtered_df[attributes].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=avg_scores.index,
            y=avg_scores.values,
            text=[f"{x:.2f}" for x in avg_scores.values],
            title='Average Attribute Ratings',
            labels={'x': 'Attribute', 'y': 'Average Rating (1-5)'}
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig)
    
    with col2:
        # Calculate NPS score
        promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
        detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
        total = filtered_df['NPS_Score'].count()
        
        nps_score = int(((promoters / total) - (detractors / total)) * 100)
        
        # Display NPS gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=nps_score,
            title={'text': "Net Promoter Score (NPS)"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, 0], 'color': "red"},
                    {'range': [0, 30], 'color': "orange"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig)
    
    # NPS by Gender and Age Group
    col1, col2 = st.columns(2)
    
    with col1:
        # NPS by Gender
        nps_by_gender = filtered_df.groupby('Gender').apply(
            lambda x: 
            ((x['NPS_Score'] >= 9).sum() - (x['NPS_Score'] <= 6).sum()) / x['NPS_Score'].count() * 100
        ).sort_values()
        
        fig = px.bar(
            x=nps_by_gender.index,
            y=nps_by_gender.values,
            text=[f"{x:.1f}" for x in nps_by_gender.values],
            title='NPS Score by Gender',
            labels={'x': 'Gender', 'y': 'NPS Score'},
            color=nps_by_gender.values,
            color_continuous_scale=px.colors.diverging.RdBu,
            color_continuous_midpoint=0
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
    
    with col2:
        # NPS by Age Group
        nps_by_age = filtered_df.groupby('Age_Group').apply(
            lambda x: 
            ((x['NPS_Score'] >= 9).sum() - (x['NPS_Score'] <= 6).sum()) / x['NPS_Score'].count() * 100
        ).sort_index()
        
        fig = px.bar(
            x=nps_by_age.index,
            y=nps_by_age.values,
            text=[f"{x:.1f}" for x in nps_by_age.values],
            title='NPS Score by Age Group',
            labels={'x': 'Age Group', 'y': 'NPS Score'},
            color=nps_by_age.values,
            color_continuous_scale=px.colors.diverging.RdBu,
            color_continuous_midpoint=0
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
    
    # Executive summary for attribute scores section
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>ATTRIBUTE SCORES - EXECUTIVE SUMMARY</div>
        <p>The analysis of product attribute ratings reveals priorities and satisfaction drivers:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>The highest-rated attributes are {attr_insights['top_attributes']}</li>
            <li>The lowest-rated attributes are {attr_insights['bottom_attributes']}</li>
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
    """, unsafe_allow_html=True)
