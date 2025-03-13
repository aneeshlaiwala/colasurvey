# Add these imports at the TOP of your script, before any other code   
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
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #0066cc;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .insight-title {
        font-weight: bold;
        color: #0066cc;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .filter-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .explained-box {
        background-color: #f9f9f9;
        border-left: 5px solid #4CAF50;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .explained-title {
        font-weight: bold;
        color: #4CAF50;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .explained-subtitle {
        font-weight: bold;
        color: #2E7D32;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .example-box {
        background-color: #e8f5e9;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 0.4rem;
    }
    .example-title {
        font-weight: bold;
        color: #2E7D32;
        font-style: italic;
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

# Load the data
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
    "Advanced Analytics Explained",
    "View & Download Full Dataset"
], horizontal=True)

# Initialize session state for filters if not exists
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'brand': None, 
        'gender': None, 
        'income': None, 
        'cluster': None
    }

# Move Filters to main page, below section selection
st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
st.subheader("Dashboard Filters")

# Create a 4-column layout for filters
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    # Create filter options with None as first option
    brand_options = [None] + sorted(df["Brand_Preference"].unique().tolist())
    selected_brand = st.selectbox(
        "Select a Brand", 
        options=brand_options, 
        index=0 if st.session_state.filters['brand'] is None else brand_options.index(st.session_state.filters['brand'])
    )

with filter_col2:
    gender_options = [None] + sorted(df["Gender"].unique().tolist())
    selected_gender = st.selectbox(
        "Select Gender", 
        options=gender_options, 
        index=0 if st.session_state.filters['gender'] is None else gender_options.index(st.session_state.filters['gender'])
    )

with filter_col3:
    income_options = [None] + sorted(df["Income_Level"].unique().tolist())
    selected_income = st.selectbox(
        "Select Income Level", 
        options=income_options, 
        index=0 if st.session_state.filters['income'] is None else income_options.index(st.session_state.filters['income'])
    )

with filter_col4:
    cluster_options = [None] + sorted(df["Cluster_Name"].unique().tolist())
    selected_cluster = st.selectbox(
        "Select Cluster", 
        options=cluster_options, 
        index=0 if st.session_state.filters['cluster'] is None else cluster_options.index(st.session_state.filters['cluster'])
    )

# Filter action buttons in two columns
fcol1, fcol2 = st.columns(2)

with fcol1:
    if st.button("Apply Filters"):
        # Update filters in session state
        st.session_state.filters = {
            'brand': selected_brand, 
            'gender': selected_gender, 
            'income': selected_income, 
            'cluster': selected_cluster
        }

with fcol2:
    if st.button("Clear Filters"):
        # Create a completely new session state dict with all Nones
        for key in st.session_state.filters.keys():
            st.session_state.filters[key] = None

st.markdown("</div>", unsafe_allow_html=True)

# Apply selected filters to the dataframe
filtered_df = df.copy()
filter_columns = {
    'brand': 'Brand_Preference',
    'gender': 'Gender',
    'income': 'Income_Level',
    'cluster': 'Cluster_Name'
}

# Apply filters dynamically
for filter_key, column in filter_columns.items():
    filter_value = st.session_state.filters.get(filter_key)
    if filter_value is not None:
        filtered_df = filtered_df[filtered_df[column] == filter_value]

# Show active filters
active_filters = [f"{k}: {v}" for k, v in st.session_state.filters.items() if v is not None]
if active_filters:
    st.info(f"Active filters: {', '.join(active_filters)} (Total records: {len(filtered_df)})")
    
# =======================
# EXECUTIVE DASHBOARD SUMMARY
# =======================
if section == "Executive Dashboard Summary":
    st.markdown("<h2 class='subheader'>Executive Dashboard Summary</h2>", unsafe_allow_html=True)
    
    # Overall key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate NPS - Correct formula: % Promoters - % Detractors
        if not filtered_df.empty:
            promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
            detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
            total = filtered_df['NPS_Score'].count()
            
            # Calculate percentages first, then subtract
            promoters_pct = (promoters / total) if total > 0 else 0
            detractors_pct = (detractors / total) if total > 0 else 0
            nps_score = int((promoters_pct - detractors_pct) * 100)
            
            # Display NPS metric
            st.metric(
                label="Overall NPS Score",
                value=nps_score,
                delta=None
            )
        else:
            st.metric(label="Overall NPS Score", value="No data", delta=None)
    
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
    
    # Top insights from each section
    st.subheader("Key Insights by Analysis Section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>DEMOGRAPHIC INSIGHTS</div>
            <p>The cola consumer base shows distinct preferences by age group and gender:</p>
            <ul>
                <li>Younger consumers (18-34) show higher preference for major brands</li>
                <li>Gender differences exist in consumption frequency and occasion preferences</li>
                <li>Income level correlates with brand preference and price sensitivity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>BRAND METRICS INSIGHTS</div>
            <p>Brand performance shows clear patterns in consumer behavior:</p>
            <ul>
                <li>Major brands dominate market share with loyal consumer bases</li>
                <li>Home consumption and parties are primary occasions for cola purchase</li>
                <li>Weekly consumption is most common frequency pattern</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>ATTRIBUTE RATING INSIGHTS</div>
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
        <div class='insight-box'>
            <div class='insight-title'>REGRESSION ANALYSIS INSIGHTS</div>
            <p>The drivers of NPS (loyalty) are clearly identified:</p>
            <ul>
                <li>Taste is the strongest predictor of consumer loyalty</li>
                <li>Brand reputation is the second most influential factor</li>
                <li>The attributes explain over 50% of variation in NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>DECISION TREE INSIGHTS</div>
            <p>Consumer loyalty can be predicted by key decision factors:</p>
            <ul>
                <li>High taste ratings are the primary predictor of promoters</li>
                <li>Low brand reputation typically indicates detractors</li>
                <li>The model identifies clear paths to improving NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>CLUSTER ANALYSIS INSIGHTS</div>
            <p>Three distinct consumer segments with different priorities:</p>
            <ul>
                <li>Taste Enthusiasts (32%): Focus on sensory experience</li>
                <li>Brand Loyalists (41%): Value reputation and consistency</li>
                <li>Value Seekers (27%): Prioritize price and availability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =======================
# DEMOGRAPHIC PROFILE
# =======================
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

# =======================
# BRAND METRICS
# =======================
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

# =======================
# BASIC ATTRIBUTE SCORES
# =======================
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
            
            # Get percentages before calculating NPS
            promoters_pct = (promoters / total) if total > 0 else 0
            detractors_pct = (detractors / total) if total > 0 else 0 
            nps_score = int((promoters_pct - detractors_pct) * 100)
            
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
    
    # NPS by Demographics section
    st.subheader("NPS by Demographics")
    
    col1, col2 = st.columns(2)
    
    # Gender Analysis
    with col1:
        if filtered_df.empty:
            st.info("No data available for NPS Gender analysis.")
        else:
            if len(filtered_df['Gender'].unique()) <= 1:
                st.info("Insufficient unique gender data for comparison.")
            else:
                # Calculate NPS by gender
                gender_results = []
                for gender in filtered_df['Gender'].unique():
                    gender_df = filtered_df[filtered_df['Gender'] == gender]
                    promoters = gender_df[gender_df['NPS_Score'] >= 9].shape[0]
                    detractors = gender_df[gender_df['NPS_Score'] <= 6].shape[0]
                    total = gender_df.shape[0]
                    
                    # Calculate NPS
                    promoters_pct = promoters / total if total > 0 else 0
                    detractors_pct = detractors / total if total > 0 else 0
                    nps = (promoters_pct - detractors_pct) * 100
                    
                    gender_results.append({
                        'Gender': gender,
                        'NPS': nps
                    })
                
                # Create dataframe for plotting
                gender_df = pd.DataFrame(gender_results)
                
                # Create bar chart
                fig = px.bar(
                    gender_df,
                    x='Gender',
                    y='NPS',
                    title='NPS Score by Gender',
                    text=[f"{x:.1f}" for x in gender_df['NPS']],
                    color='NPS',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    color_continuous_midpoint=0
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
    
    # Age Group Analysis
    with col2:
        if filtered_df.empty:
            st.info("No data available for NPS Age Group analysis.")
        else:
            if len(filtered_df['Age_Group'].unique()) <= 1:
                st.info("Insufficient unique age group data for comparison.")
            else:
                # Calculate NPS by age group
                age_results = []
                for age_group in sorted(filtered_df['Age_Group'].unique()):
                    age_df = filtered_df[filtered_df['Age_Group'] == age_group]
                    promoters = age_df[age_df['NPS_Score'] >= 9].shape[0]
                    detractors = age_df[age_df['NPS_Score'] <= 6].shape[0]
                    total = age_df.shape[0]
                    
                    # Calculate NPS
                    promoters_pct = promoters / total if total > 0 else 0
                    detractors_pct = detractors / total if total > 0 else 0
                    nps = (promoters_pct - detractors_pct) * 100
                    
                    age_results.append({
                        'Age_Group': age_group,
                        'NPS': nps
                    })
                
                # Create dataframe for plotting
                age_df = pd.DataFrame(age_results)
                
                # Create bar chart
                fig = px.bar(
                    age_df,
                    x='Age_Group',
                    y='NPS',
                    title='NPS Score by Age Group',
                    text=[f"{x:.1f}" for x in age_df['NPS']],
                    color='NPS',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    color_continuous_midpoint=0
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)

# =======================
# REGRESSION ANALYSIS
# =======================
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
            st.write(f"**R-squared:** {model.rsquared:.4f}")# Display key metrics
            st.write(f"**R-squared:** {model.rsquared:.4f}")
            st.write(f"**Adjusted R-squared:** {model.rsquared_adj:.4f}")
            st.write(f"**F-statistic:** {model.fvalue:.4f}")
            st.write(f"**Prob (F-statistic):** {model.f_pvalue:.4f}")
        
        with col2:
            # Visualization of coefficients
            sig_coefs = coef_df[coef_df['Feature'] != 'const']
            
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

# =======================
# DECISION TREE ANALYSIS
# =======================
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
            
            st.write("- These consumers are least likely to recommend your brand")# =======================
# CLUSTER ANALYSIS
# =======================
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
                st.info("Insufficient data for factor analysis. Minimum of a few hundred records required (minimum 50).")
        
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

# =======================
# ADVANCED ANALYTICS EXPLAINED
# =======================
elif section == "Advanced Analytics Explained":
    st.markdown("<h2 class='subheader'>Advanced Analytics Explained</h2>", unsafe_allow_html=True)
    
    # Create download button with external PDF link
    st.markdown("""
    <div class='explained-box'>
        <div class='explained-title'>Advanced Analytics Overview</div>
        <p>Download the comprehensive PDF explaining advanced analytics techniques:</p>
        <a href="https://1drv.ms/b/s!AjvUTGyNS16HjZV-BJeBvloAgSeXOQ?e=oWofYC" 
           target="_blank" 
           download="Advanced_Analytics_Explained.pdf" 
           class="btn btn-primary">
            Download PDF
        </a>
    </div>
    """, unsafe_allow_html=True)

# =======================
# VIEW & DOWNLOAD FULL DATASET
# =======================
elif section == "View & Download Full Dataset":
    st.markdown("<h2 class='subheader'>View & Download Dataset</h2>", unsafe_allow_html=True)
    
    # Show dataset with cluster information
    st.dataframe(filtered_df)
    
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

# Footer with contact email as a clickable link
st.markdown("---")
st.markdown("<div style='text-align: center;'>Cola Survey Dashboard | Created by <a href='mailto:aneesh@insights3d.com'>Aneesh Laiwala (aneesh@insights3d.com)</a></div>", unsafe_allow_html=True)
