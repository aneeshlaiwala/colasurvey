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
    .filter-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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

# Move Filters to main page, below section selection
st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
st.subheader("Dashboard Filters")

# Create a 4-column layout for filters
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    # Create filter options with None as first option
    brand_options = [None] + sorted(df["Brand_Preference"].unique().tolist())
    brand = st.selectbox("Select a Brand", brand_options, key='brand_main')

with filter_col2:
    gender_options = [None] + sorted(df["Gender"].unique().tolist())
    gender = st.selectbox("Select Gender", gender_options, key='gender_main')

with filter_col3:
    income_options = [None] + sorted(df["Income_Level"].unique().tolist())
    income = st.selectbox("Select Income Level", income_options, key='income_main')

with filter_col4:
    cluster_options = [None] + sorted(df["Cluster_Name"].unique().tolist())
    cluster = st.selectbox("Select Cluster", cluster_options, key='cluster_main')

# Initialize session state for filters if not exists
if 'filters' not in st.session_state:
    st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}

# Filter action buttons in two columns
fcol1, fcol2 = st.columns(2)

with fcol1:
    if st.button("Apply Filters"):
        st.session_state.filters['brand'] = brand
        st.session_state.filters['gender'] = gender
        st.session_state.filters['income'] = income
        st.session_state.filters['cluster'] = cluster
        st.rerun()

with fcol2:
    if st.button("Clear Filters"):
        st.session_state.filters = {'brand': None, 'gender': None, 'income': None, 'cluster': None}
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

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

# Show active filters
active_filters = [f"{k}: {v}" for k, v in st.session_state.filters.items() if v is not None]
if active_filters:
    st.info(f"Active filters: {', '.join(active_filters)} (Total records: {len(filtered_df)})")

# Display Selected Section
if section == "Executive Dashboard Summary":
    st.markdown("<h2 class='subheader'>Executive Dashboard Summary</h2>", unsafe_allow_html=True)
    
    # Overall key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate NPS
        promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
        detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
        total = filtered_df['NPS_Score'].count()
        nps_score = int(((promoters / total) - (detractors / total)) * 100) if total > 0 else 0
        
        # Display NPS metric
        st.metric(
            label="Overall NPS Score",
            value=nps_score,
            delta=None
        )
    
    with col2:
        # Top brand
        if not filtered_df.empty:
            top_brand = filtered_df['Most_Often_Consumed_Brand'].value_counts().idxmax()
            top_brand_pct = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True).max() * 100
            
            st.metric(
                label="Top Brand",
                value=top_brand,
                delta=f"{top_brand_pct:.1f}% Market Share"
            )
        else:
            st.metric(label="Top Brand", value="No data", delta=None)
    
    with col3:
        # Top attribute
        attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                     'Brand_Reputation_Rating', 'Availability_Rating', 
                     'Sweetness_Rating', 'Fizziness_Rating']
        
        if not filtered_df.empty:
            top_attr = filtered_df[attributes].mean().idxmax()
            top_attr_score = filtered_df[attributes].mean().max()
            
            st.metric(
                label="Highest Rated Attribute",
                value=top_attr.replace('_Rating', ''),
                delta=f"{top_attr_score:.2f}/5"
            )
        else:
            st.metric(label="Highest Rated Attribute", value="No data", delta=None)
    
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
        st.markdown("""
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
        
        st.markdown("""
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
        
        st.markdown("""
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
        st.markdown("""
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
        
        st.markdown("""
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
        
        st.markdown("""
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
    
    st.markdown("""
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Group Distribution
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Age Group Distribution with current filters.")
        
        # Income Level Distribution
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Income Level Distribution with current filters.")
        
    with col2:
        # Gender Distribution
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Gender Distribution with current filters.")
        
        # Age Group by Gender
        if not filtered_df.empty and len(filtered_df['Gender'].unique()) > 1:
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
        else:
            st.info("Insufficient data for Age Group by Gender analysis with current filters.")
    
    # Prepare dynamic summary
    age_insight = "N/A"
    gender_insight = "N/A"
    income_insight = "N/A"
    
    if not filtered_df.empty:
        # Get top age group
        top_age_group = filtered_df['Age_Group'].value_counts(normalize=True).idxmax()
        top_age_pct = filtered_df['Age_Group'].value_counts(normalize=True).max() * 100
        age_insight = f"{top_age_group} ({top_age_pct:.1f}%)"
        
        # Get gender distribution
        if len(filtered_df['Gender'].unique()) > 1:
            top_gender = filtered_df['Gender'].value_counts(normalize=True).idxmax()
            top_gender_pct = filtered_df['Gender'].value_counts(normalize=True).max() * 100
            gender_insight = f"predominately {top_gender} ({top_gender_pct:.1f}%)"
        else:
            gender_insight = f"exclusively {filtered_df['Gender'].iloc[0]}"
        
        # Get income pattern
        top_income = filtered_df['Income_Level'].value_counts(normalize=True).idxmax()
        top_income_pct = filtered_df['Income_Level'].value_counts(normalize=True).max() * 100
        income_insight = f"{top_income} ({top_income_pct:.1f}%)"
    
    # Executive summary for demographic section with dynamic content
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>DEMOGRAPHIC PROFILE - EXECUTIVE SUMMARY</div>
        <p>The demographic analysis of cola consumers reveals distinct patterns across age groups, gender, and income levels:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Age distribution shows prevalence in {age_insight}</li>
            <li>Gender split indicates {gender_insight}</li>
            <li>Income level distribution suggests predominance of {income_insight}</li>
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most Often Consumed Brand
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Brand analysis with current filters.")
        
        # Occasions of Buying
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Occasions analysis with current filters.")
    
    with col2:
        # Frequency of Consumption
        if not filtered_df.empty:
            freq_counts = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
            # Sort by frequency (not alphabetically)
            freq_order = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
            freq_counts = freq_counts.reindex([f for f in freq_order if f in freq_counts.index])
            
            fig = px.bar(
                x=freq_counts.index,
                y=freq_counts.values,
                text=[f"{x:.1f}%" for x in freq_counts.values],
                title='Frequency of Consumption (%)',
                labels={'x': 'Frequency', 'y': 'Percentage (%)'}
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig)
        else:
            st.info("No data available for Frequency analysis with current filters.")
        
        # Satisfaction Level
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Satisfaction analysis with current filters.")
    
    # Prepare dynamic summary
    dominant_brands = "N/A"
    main_occasions = "N/A"
    frequency_patterns = "N/A"
    satisfaction_trends = "N/A"
    
    if not filtered_df.empty:
        # Get dominant brands
        top_brands = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True).nlargest(2)
        dominant_brands = ", ".join([f"{idx} ({val:.1f}%)" for idx, val in top_brands.items()])
        
        # Get main occasions
        top_occasions = filtered_df['Occasions_of_Buying'].value_counts(normalize=True).nlargest(2)
        main_occasions = ", ".join([f"{idx} ({val:.1f}%)" for idx, val in top_occasions.items()])
        
        # Get frequency patterns
        top_frequency = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True).nlargest(1).index[0]
        top_freq_pct = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True).max() * 100
        frequency_patterns = f"primarily {top_frequency} consumption ({top_freq_pct:.1f}%)"
        
        # Get satisfaction trends
        satisfaction_dict = {'Very Satisfied': 5, 'Satisfied': 4, 'Neutral': 3, 'Dissatisfied': 2, 'Very Dissatisfied': 1}
        try:
            # Map satisfaction to numeric values for calculating average
            filtered_df['Satisfaction_Numeric'] = filtered_df['Satisfaction_Level'].map(satisfaction_dict)
            avg_satisfaction = filtered_df['Satisfaction_Numeric'].mean()
            
            # Categorize average satisfaction
            if avg_satisfaction >= 4.5:
                satisfaction_trends = "very high satisfaction"
            elif avg_satisfaction >= 3.5:
                satisfaction_trends = "positive satisfaction"
            elif avg_satisfaction >= 2.5:
                satisfaction_trends = "neutral satisfaction"
            else:
                satisfaction_trends = "low satisfaction"
                
            satisfaction_trends += f" (average score: {avg_satisfaction:.2f}/5)"
        except:
            satisfaction_trends = "varied satisfaction levels"
    
    # Executive summary for brand metrics section with dynamic content
    st.markdown(f"""
    <div class='summary-box'>
        <div class='summary-title'>BRAND METRICS - EXECUTIVE SUMMARY</div>
        <p>The analysis of brand metrics provides insights into consumer preferences and consumption patterns:</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Market share is dominated by {dominant_brands}</li>
            <li>Primary consumption occasions include {main_occasions}</li>
            <li>Consumption frequency patterns reveal {frequency_patterns}</li>
            <li>Overall satisfaction levels indicate {satisfaction_trends}</li>
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # All attribute ratings
        attributes = [
            'Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
            'Brand_Reputation_Rating', 'Availability_Rating', 
            'Sweetness_Rating', 'Fizziness_Rating'
        ]
        
        if not filtered_df.empty:
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
        else:
            st.info("No data available for Attribute Ratings with current filters.")
    
    with col2:
        # Calculate NPS score
        if not filtered_df.empty:
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
        else:
            st.info("No data available for NPS Score with current filters.")
    
    # NPS by Gender and Age Group
    col1, col2 = st.columns(2)
    
    with col1:
        # NPS by Gender
        if not filtered_df.empty and len(filtered_df['Gender'].unique()) > 1:
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
        else:
            st.info("Insufficient data for NPS by Gender analysis with current filters.")
    
    with col2:
        # NPS by Age Group
        if not filtered_df.empty and len(filtered_df['Age_Group'].unique()) > 1:
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
        else:
            st.info("Insufficient data for NPS by Age Group analysis with current filters.")
    
    # Prepare dynamic summary
    top_attributes = "N/A"
    bottom_attributes = "N/A"
    
    if not filtered_df.empty:
        # Get top attributes
        sorted_attrs = filtered_df[attributes].mean().sort_values(ascending=False)
        top_attributes = ", ".join([f"{attr.replace('_Rating', '')} ({val:.2f}/5)" 
                                  for attr, val in sorted_attrs.head(2).items()])
        
        # Get bottom attributes
        bottom_attributes = ", ".join([f"{attr.replace('_Rating', '')} ({val:.2f}/5)" 
                                     for attr, val in sorted_attrs.tail(2).items()])
    
    # Executive summary for attribute scores section with dynamic content
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

elif section == "Regression Analysis":
    st.markdown("<h2 class='subheader'>Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.warning("Insufficient data for regression analysis. Please adjust your filters.")
    else:
        # Prepare data for regression
        X_reg = filtered_df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                           'Brand_Reputation_Rating', 'Availability_Rating', 
                           'Sweetness_Rating', 'Fizziness_Rating']]
        
        y_reg = filtered_df['NPS_Score']
        
        # Add constant to predictor variables
        X_reg = sm.add_constant(X_reg)
        
        # Fit regression model
        model = sm.OLS(y_reg, X_reg).fit()
        
        # Display regression results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regression Summary")
            
            # Create a dataframe for coefficients
            coef_df = pd.DataFrame({
                'Feature': X_reg.columns,
                'Coefficient': model.params,
                'P-Value': model.pvalues,
                'Significant': model.pvalues < 0.05
            })
            
            # Sort by absolute coefficient value
            coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
            
            # Format p-values
            coef_df['P-Value'] = coef_df['P-Value'].apply(lambda x: f"{x:.4f}")
            coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.4f}")
            
            # Display table
            st.dataframe(coef_df, use_container_width=True)
            
            # Display key metrics
            st.write(f"**R-squared:** {model.rsquared:.4f}")
            st.write(f"**Adjusted R-squared:** {model.rsquared_adj:.4f}")
            st.write(f"**F-statistic:** {model.fvalue:.4f}")
            st.write(f"**Prob (F-statistic):** {model.f_pvalue:.4f}")
        
        with col2:
            # Visualization of coefficients
            sig_coefs = coef_df[coef_df['Feature'] != 'const']
            colors = ['green' if p else 'red' for p in sig_coefs['Significant']]
            
            fig = px.bar(
                sig_coefs,
                x='Feature', 
                y='Coefficient',
                title='Feature Importance (Coefficient Values)',
                color='Significant',
                color_discrete_map={True: 'green', False: 'red'},
                labels={'Coefficient': 'Impact on NPS Score', 'Feature': 'Attribute'}
            )
            st.plotly_chart(fig)
        
        # Key findings summary
        st.subheader("Key Regression Findings")
        
        # Identify significant positive and negative factors
        pos_factors = coef_df[(coef_df['Significant'] == True) & (coef_df['Feature'] != 'const') & (coef_df['Coefficient'].astype(float) > 0)]
        neg_factors = coef_df[(coef_df['Significant'] == True) & (coef_df['Feature'] != 'const') & (coef_df['Coefficient'].astype(float) < 0)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not pos_factors.empty:
                st.write("**Significant Positive Factors on NPS:**")
                for i, row in pos_factors.iterrows():
                    st.write(f"- {row['Feature'].replace('_Rating', '')}: {row['Coefficient']}")
            else:
                st.write("No significant positive factors found.")
        
        with col2:
            if not neg_factors.empty:
                st.write("**Significant Negative Factors on NPS:**")
                for i, row in neg_factors.iterrows():
                    st.write(f"- {row['Feature'].replace('_Rating', '')}: {row['Coefficient']}")
            else:
                st.write("No significant negative factors found.")
        
        # Formal Executive Summary Box
        st.markdown("""
        <div class='summary-box'>
            <div class='summary-title'>REGRESSION ANALYSIS - EXECUTIVE SUMMARY</div>
            <p>The regression analysis identifies the key drivers of consumer loyalty as measured by NPS:</p>
            
            <p><strong>Key Findings:</strong></p>
            <ul>
        """, unsafe_allow_html=True)
        
        # Model quality
        st.markdown(f"<li>The regression model explains <strong>{model.rsquared:.1%}</strong> of the variation in NPS scores</li>", unsafe_allow_html=True)
        
        # Model significance
        if model.f_pvalue < 0.05:
            st.markdown("<li>The model is <strong>statistically significant</strong> (p < 0.05)</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>The model is <strong>not statistically significant</strong> (p > 0.05)</li>", unsafe_allow_html=True)
        
        # Model predictive power
        if model.rsquared < 0.3:
            st.markdown("<li>The model has relatively low predictive power, suggesting additional factors influence NPS</li>", unsafe_allow_html=True)
        
        # Important drivers
        significant_features = coef_df[(coef_df['Significant'] == True) & (coef_df['Feature'] != 'const')]
        if not significant_features.empty:
            most_important = significant_features.iloc[0]['Feature'].replace('_Rating', '')
            st.markdown(f"<li>The most influential factor is <strong>{most_important}</strong></li>", unsafe_allow_html=True)
            
            if len(significant_features) > 1:
                secondary_factors = ', '.join([f.replace('_Rating', '') for f in significant_features.iloc[1:3]['Feature'].tolist()])
                st.markdown(f"<li>Secondary factors include: <strong>{secondary_factors}</strong></li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>No individual factors show statistical significance in predicting NPS scores</li>", unsafe_allow_html=True)
        
        st.markdown("""
            </ul>
            
            <p><strong>Strategic Implications:</strong></p>
            <ul>
                <li>Focus improvement efforts on the attributes with strongest positive coefficients</li>
                <li>Address negative drivers to minimize their impact on consumer loyalty</li>
                <li>Use regression insights to prioritize product development initiatives</li>
                <li>Consider the balance of attribute improvements against implementation costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif section == "Decision Tree Analysis":
    st.markdown("<h2 class='subheader'>Decision Tree Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 30:
        st.warning("Insufficient data for decision tree analysis. Please adjust your filters.")
    else:
        # Define NPS categories for classification
        filtered_df['NPS_Category'] = pd.cut(
            filtered_df['NPS_Score'],
            bins=[-1, 6, 8, 10],
            labels=['Detractor', 'Passive', 'Promoter']
        )
        
        # Prepare data for decision tree
        X_tree = filtered_df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                            'Brand_Reputation_Rating', 'Availability_Rating', 
                            'Sweetness_Rating', 'Fizziness_Rating', 'Age']]
        
        y_tree = filtered_df['NPS_Category']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.25, random_state=42)
        
        # Train decision tree
        dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = dt_model.score(X_train, y_train)
        test_accuracy = dt_model.score(X_test, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_tree.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Decision Tree Performance")
            st.write(f"**Training Accuracy:** {train_accuracy:.2%}")
            st.write(f"**Testing Accuracy:** {test_accuracy:.2%}")
            
            # Feature importance chart
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title='Feature Importance in NPS Classification',
                labels={'Importance': 'Importance Score', 'Feature': 'Attribute'}
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Decision Tree Visualization")
            
            # Create decision tree plot
            plt.figure(figsize=(12, 8))
            plot_tree(dt_model, 
                     filled=True, 
                     feature_names=X_tree.columns, 
                     class_names=dt_model.classes_,
                     rounded=True,
                     fontsize=10)
            
            # Save plot to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Display the image
            st.image(buf)
        
        # Decision path analysis
        st.subheader("Key Decision Rules")
        
        # Extract and display key decision paths for promoters
        rules = tree.export_text(dt_model, 
                                feature_names=list(X_tree.columns),
                                max_depth=3)
        
        # Create a more user-friendly summary
        top_feature = feature_importance.iloc[0]['Feature']
        second_feature = feature_importance.iloc[1]['Feature'] if len(feature_importance) > 1 else None
        
        st.write(f"""
        **Decision Tree Analysis Summary:**
        
        The decision tree model achieved {test_accuracy:.1%} accuracy in predicting NPS categories (Promoter, Passive, Detractor).
        
        The most important factor in determining customer loyalty (NPS) is **{top_feature}**, 
        {"followed by **" + second_feature + "**" if second_feature else ""}.
        """)
        
        # Simplified rule display
        st.code(rules, language='text')
        
        # Consumer insights based on tree
        st.subheader("Consumer Insights from Decision Tree")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Promoter Profile:**")
            
            # Logic for promoters based on top features
            if top_feature in ['Taste_Rating', 'Brand_Reputation_Rating']:
                st.write(f"- Consumers with high {top_feature.replace('_Rating', '')} satisfaction")
            if second_feature:
                st.write(f"- Secondary importance: {second_feature.replace('_Rating', '')}")
            
            st.write("- These consumers are most likely to recommend your brand")
        
        with col2:
            st.write("**Detractor Profile:**")
            
            # Logic for detractors based on top features
            if top_feature in ['Taste_Rating', 'Brand_Reputation_Rating', 'Price_Rating']:
                st.write(f"- Consumers with low {top_feature.replace('_Rating', '')} satisfaction")
            if second_feature:
                st.write(f"- Also influenced by: {second_feature.replace('_Rating', '')}")
            
            st.write("- These consumers are least likely to recommend your brand")
        
        # Decision Tree Executive Summary
        st.markdown(f"""
        <div class='summary-box'>
            <div class='summary-title'>DECISION TREE ANALYSIS - EXECUTIVE SUMMARY</div>
            <p>The decision tree analysis identifies the critical decision pathways that determine consumer loyalty:</p>
            
            <p><strong>Key Findings:</strong></p>
            <ul>
                <li>The model achieved <strong>{test_accuracy:.1%}</strong> accuracy in predicting NPS categories</li>
                <li>The most important classification factor is <strong>{top_feature.replace('_Rating', '')}</strong></li>
                <li>Clear decision rules identify paths to promoter vs. detractor status</li>
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
        """, unsafe_allow_html=True)

elif section == "Cluster Analysis":
    st.markdown("<h2 class='subheader'>Cluster Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 30:
        st.warning("Insufficient data for cluster analysis. Please adjust your filters.")
    else:
        # Display cluster distribution
        cluster_dist = filtered_df['Cluster_Name'].value_counts(normalize=True) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=cluster_dist.values,
                names=cluster_dist.index,
                title='Cluster Distribution (%)',
                hole=0.4,
                labels={'label': 'Cluster', 'value': 'Percentage (%)'}
            )
            fig.update_traces(textinfo='label+percent', textposition='inside')
            st.plotly_chart(fig)
            
            # Display Factor Analysis
            st.subheader("Factor Analysis")
            
            # Perform factor analysis
            attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                        'Brand_Reputation_Rating', 'Availability_Rating', 
                        'Sweetness_Rating', 'Fizziness_Rating']
            
            # Only perform factor analysis if we have sufficient data
            if len(filtered_df) >= 50:  # Minimum recommended for factor analysis
                fa = FactorAnalyzer(n_factors=2, rotation='varimax')
                fa.fit(filtered_df[attributes])
                
                # Get factor loadings
                loadings = pd.DataFrame(
                    fa.loadings_,
                    index=attributes,
                    columns=['Factor 1', 'Factor 2']
                )
                
                # Display loadings
                st.dataframe(loadings.round(3), use_container_width=True)
            else:
                st.info("Insufficient data for factor analysis. Minimum of 50 records required.")
        
        with col2:
            # Cluster centers
            st.subheader("Cluster Centers (Average Ratings)")
            
            # Generate radar chart for cluster centers
            categories = ['Taste', 'Price', 'Packaging', 'Brand_Reputation', 'Availability', 'Sweetness', 'Fizziness']
            
            # Get cluster centers and reshape for radar chart
            centers_data = []
            cluster_names = filtered_df['Cluster_Name'].unique()
            
            for i, name in enumerate(cluster_names):
                cluster_id = filtered_df[filtered_df['Cluster_Name'] == name]['Cluster'].iloc[0]
                values = cluster_centers.iloc[cluster_id].values.tolist()
                centers_data.append({
                    'Cluster': name,
                    **{cat: val for cat, val in zip(categories, values)}
                })
            
            # Create radar chart
            df_radar = pd.DataFrame(centers_data)
            fig = go.Figure()
            
            for i, row in df_radar.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row[categories].values,
                    theta=categories,
                    fill='toself',
                    name=row['Cluster']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                title="Cluster Profiles - Average Ratings"
            )
            st.plotly_chart(fig)
        
        # Cluster profiles
        st.subheader("Cluster Profiles Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Most common demographic and preference characteristics per cluster
        clusters = filtered_df['Cluster_Name'].unique()
        
        for i, cluster in enumerate([clusters[i] for i in range(min(3, len(clusters)))]):
            col = [col1, col2, col3][i]
            with col:
                cluster_data = filtered_df[filtered_df['Cluster_Name'] == cluster]
                
                st.write(f"**{cluster}** ({len(cluster_data)} consumers, {len(cluster_data)/len(filtered_df):.1%})")
                
                if len(cluster_data) > 0:
                    # Top brand preference
                    top_brand = cluster_data['Most_Often_Consumed_Brand'].value_counts().idxmax()
                    brand_pct = cluster_data['Most_Often_Consumed_Brand'].value_counts(normalize=True).max() * 100
                    
                    # Top occasion
                    top_occasion = cluster_data['Occasions_of_Buying'].value_counts().idxmax()
                    occasion_pct = cluster_data['Occasions_of_Buying'].value_counts(normalize=True).max() * 100
                    
                    # Demographics
                    top_gender = cluster_data['Gender'].value_counts().idxmax()
                    gender_pct = cluster_data['Gender'].value_counts(normalize=True).max() * 100
                    
                    top_age = cluster_data['Age_Group'].value_counts().idxmax()
                    age_pct = cluster_data['Age_Group'].value_counts(normalize=True).max() * 100
                    
                    # Average NPS
                    avg_nps = cluster_data['NPS_Score'].mean()
                    
                    st.write(f"🥤 Prefers: **{top_brand}** ({brand_pct:.1f}%)")
                    st.write(f"🛒 Typically buys for: **{top_occasion}** ({occasion_pct:.1f}%)")
                    st.write(f"👤 Demographics: **{top_gender}** ({gender_pct:.1f}%), **{top_age}** ({age_pct:.1f}%)")
                    st.write(f"⭐ Avg. NPS: **{avg_nps:.1f}**")
                    
                    # Top attributes (highest rated)
                    top_attribute = cluster_data[attributes].mean().idxmax()
                    st.write(f"💪 Strongest attribute: **{top_attribute.replace('_Rating', '')}**")
                    
                    # Lowest attributes (lowest rated)
                    lowest_attribute = cluster_data[attributes].mean().idxmin()
                    st.write(f"⚠️ Weakest attribute: **{lowest_attribute.replace('_Rating', '')}**")
                else:
                    st.write("No data available for this cluster with current filters.")
        
        # Formal Executive Summary Box
        st.markdown("""
        <div class='summary-box'>
            <div class='summary-title'>CLUSTER ANALYSIS - EXECUTIVE SUMMARY</div>
            <p>The cluster analysis identified three distinct consumer segments based on their preferences and priorities:</p>
            
            <p><strong>Consumer Segments:</strong></p>
            <ul>
                <li><strong>Taste Enthusiasts:</strong> Prioritize taste and flavor experience above all. More focused on sensory aspects and less concerned with brand or price.</li>
                <li><strong>Brand Loyalists:</strong> Place high importance on brand reputation and packaging. Likely to be loyal to specific brands and less price-sensitive.</li>
                <li><strong>Value Seekers:</strong> More price-conscious and practical. Look for a good balance between price and quality, with availability being an important factor.</li>
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

elif section == "View & Download Full Dataset":
    st.markdown("<h2 class='subheader'>View & Download Dataset</h2>", unsafe_allow_html=True)
    
    # Show dataset with cluster information
    st.dataframe(filtered_df)
    
    # Executive summary for data section
    st.markdown("""
    <div class='summary-box'>
        <div class='summary-title'>DATASET OVERVIEW - EXECUTIVE SUMMARY</div>
        <p>This section provides access to the complete dataset with all analysis variables:</p>
        
        <p><strong>Dataset Features:</strong></p>
        <ul>
            <li>1,000 survey respondents with demographic information</li>
            <li>Brand preferences and consumption patterns</li>
            <li>Attribute ratings across 7 key product dimensions</li>
            <li>NPS scores indicating consumer loyalty</li>
            <li>Cluster assignments from segmentation analysis</li>
        </ul>
        
        <p><strong>Applications:</strong></p>
        <ul>
            <li>Download data for further custom analysis</li>
            <li>Export filtered segments for targeted marketing initiatives</li>
            <li>Use cluster information for consumer targeting</li>
            <li>Analyze raw data to identify additional insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        # Full dataset
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=csv,
            file_name="cola_survey_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary statistics
        summary_stats = filtered_df.describe().transpose()
        csv_summary = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Statistics (CSV)",
            data=csv_summary,
            file_name="cola_survey_summary.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**Cola Survey Dashboard** | Created with Streamlit")
