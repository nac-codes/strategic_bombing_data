import streamlit as st
import pandas as pd
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="USAAF Bombing Campaign Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
PLOT_PATH = "plots/usaaf"
ORIGINAL_DATA_PATH = "combined_attack_data.csv"
PROCESSED_DATA_PATH = "processed_data/usaaf/usaaf_raids_full.csv"

# Function to load an image
def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Function to create a download link for data
def get_download_link(file_path, link_text):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link for {file_path}: {e}")
        return None

# Function to get filtered dataframe based on selection
def get_filtered_df(df, filter_type, filter_value):
    if filter_type == "city":
        # Convert filter value to uppercase to match normalized location names
        return df[df["target_location"] == filter_value.upper()]
    elif filter_type == "category":
        return df[df["CATEGORY"] == filter_value]
    elif filter_type == "year":
        return df[df["Year"] == filter_value]
    return df

# Function to format data table display
def display_data_table(df, num_rows=10):
    if len(df) > 0:
        # Create expandable section with data table
        with st.expander("View Raid Data Table", expanded=False):
            st.markdown(f"**Showing {min(num_rows, len(df))} of {len(df)} raids**")
            
            # Add a search filter
            search_term = st.text_input("Search in target name:", "")
            if search_term:
                filtered_df = df[df['target_name'].str.contains(search_term, case=False, na=False)]
            else:
                filtered_df = df
                
            # Display table with pagination
            start_idx = st.number_input("Starting row", min_value=0, max_value=max(0, len(filtered_df)-num_rows), value=0, step=num_rows)
            end_idx = min(start_idx + num_rows, len(filtered_df))
            
            # Display subset of dataframe
            st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
            
            # Calculate summary statistics
            st.markdown("### Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Area Bombing Score", f"{filtered_df['AREA_BOMBING_SCORE_NORMALIZED'].mean():.2f}")
            with col2:
                st.metric("Average Tonnage", f"{filtered_df['TOTAL_TONS'].mean():.2f}")
            with col3:
                st.metric("Average Incendiary %", f"{filtered_df['INCENDIARY_PERCENT'].mean():.2f}%")
            
            # Add download option for filtered data
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_raids.csv">Download Filtered Data</a>'
            st.markdown(href, unsafe_allow_html=True)

# Load the data for filtering options
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(PROCESSED_DATA_PATH)
        # Add Year column if it doesn't exist
        if 'Year' not in data.columns and 'YEAR' in data.columns:
            data['Year'] = (1940 + data['YEAR'].astype(float)).astype(int)
            # Handle any outlier years
            data.loc[data['Year'] < 1939, 'Year'] = 1940
            data.loc[data['Year'] > 1946, 'Year'] = 1945
        
        # Normalize location names to uppercase to prevent duplicates with different case
        data['target_location'] = data['target_location'].str.strip().str.upper()
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to get list of cities with available visualizations
def get_cities_with_visualizations():
    cities_with_viz = []
    try:
        for city in cities:
            city_lower = city.lower()
            if os.path.exists(f"{PLOT_PATH}/cities/score_distribution_city_{city_lower}.png") or \
               os.path.exists(f"{PLOT_PATH}/cities/tonnage_vs_incendiary_city_{city_lower}.png") or \
               os.path.exists(f"{PLOT_PATH}/cities/scores_by_target_type_city_{city_lower}.png") or \
               os.path.exists(f"{PLOT_PATH}/cities/category_pie_city_{city_lower}.png"):
                cities_with_viz.append((city, True))
            else:
                cities_with_viz.append((city, False))
        return cities_with_viz
    except Exception as e:
        st.error(f"Error checking for city visualizations: {e}")
        return [(city, False) for city in cities]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["General Analysis", "City Analysis", "Category Analysis", "Year Analysis", "Data Download"]
)

# Load data for filters
try:
    df = load_data()
    cities = sorted(df["target_location"].unique())
    categories = sorted(df["CATEGORY"].unique())
    years = sorted([y for y in range(1940, 1946)])
except Exception as e:
    st.sidebar.error(f"Error loading filter options: {e}")
    cities = []
    categories = []
    years = []

# Add data quality acknowledgment in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Data Quality Note
This data was derived from historical bombing records and has some limitations:
- Contains OCR errors from original documents
- May include misreadings of original records
- Could have processing/interpretation errors

Despite these limitations, the dataset provides a robust overall picture of USAAF bombing campaigns.
""")

# Main content based on navigation
if page == "General Analysis":
    st.title("USAAF Bombing Campaign Analysis")
    st.markdown("""
    This dashboard presents analysis of the United States Army Air Forces (USAAF) bombing campaigns during World War II,
    with a focus on the evolution of precision vs. area bombing strategies.
    """)
    
    # Display general visualizations stacked vertically
    st.subheader("Overall Distribution of Area Bombing Scores")
    img = load_image(f"{PLOT_PATH}/general/overall_score_distribution.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Distribution showing how USAAF missions scored on the area bombing scale (0=precise, 10=area)*")
    
    st.subheader("Tonnage Distribution")
    img = load_image(f"{PLOT_PATH}/general/tonnage_distribution.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Distribution of bombing tonnage per raid across all USAAF missions*")
    
    st.subheader("HE vs Incendiary Bombing by Year")
    img = load_image(f"{PLOT_PATH}/general/he_vs_incendiary_by_year.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Annual comparison of high explosive vs. incendiary bombing tonnage*")
    
    st.subheader("Relationship Between Tonnage and Area Bombing")
    img = load_image(f"{PLOT_PATH}/general/tonnage_vs_score_relationship.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Correlation between bombing tonnage and area bombing scores*")
    
    st.subheader("Bombing Patterns Over Time")
    img = load_image(f"{PLOT_PATH}/general/quarterly_metrics_evolution.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Quarterly evolution of key bombing metrics throughout the war*")
        
    st.subheader("Target Category Analysis")
    img = load_image(f"{PLOT_PATH}/general/he_vs_incendiary_by_category.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*HE vs. incendiary bombing by target category with area bombing scores*")
    
    img = load_image(f"{PLOT_PATH}/general/tonnage_distribution_by_category.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Distribution of bombing tonnage across different target categories*")
    
    st.subheader("Temporal Analysis by Target Category")
    img = load_image(f"{PLOT_PATH}/general/year_category_score_heatmap.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Evolution of area bombing scores by target category and year*")
        
    st.subheader("Monthly Progression")
    img = load_image(f"{PLOT_PATH}/general/monthly_score_progression.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Monthly progression of area bombing scores with marker size representing tonnage*")
    
    # Add the category by year visualization
    st.subheader("Distribution of Bombing Categories by Year")
    img = load_image(f"{PLOT_PATH}/years/category_by_year.png")
    if img:
        st.image(img, use_container_width=True)
        st.markdown("*Evolution of bombing categories throughout the war years*")
    
    # Add complete dataset view
    st.markdown("---")
    st.subheader("Explore Complete Dataset")
    display_data_table(df)

elif page == "City Analysis":
    st.title("City-Specific Bombing Analysis")
    
    # Get cities with available visualizations
    cities_with_viz = get_cities_with_visualizations()
    
    # Create a dropdown with clear indicators of which cities have visualizations
    city_options = [f"{city} {'✓' if has_viz else '○'}" for city, has_viz in cities_with_viz]
    selected_city_option = st.selectbox(
        "Select a city (✓ = visualizations available, ○ = limited or no visualizations):", 
        city_options
    )
    
    # Extract just the city name from the selected option
    selected_city = selected_city_option.split(' ')[0] if selected_city_option else None
    
    if selected_city:
        st.markdown(f"## Bombing Analysis for {selected_city}")
        
        # Display city-specific visualizations stacked vertically
        # Score distribution
        score_dist_path = f"{PLOT_PATH}/cities/score_distribution_city_{selected_city.lower()}.png"
        if os.path.exists(score_dist_path):
            img = load_image(score_dist_path)
            if img:
                st.subheader("Area Bombing Score Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of area bombing scores for raids targeting {selected_city}*")
        else:
            st.info(f"No score distribution visualization available for {selected_city}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/cities/tonnage_vs_incendiary_city_{selected_city.lower()}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img, use_container_width=True)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage for {selected_city} raids*")
        else:
            st.info(f"No tonnage visualization available for {selected_city}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/cities/scores_by_target_type_city_{selected_city.lower()}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img, use_container_width=True)
                st.markdown(f"*Breakdown of area bombing scores by target type within {selected_city}*")
        else:
            st.info(f"No target type visualization available for {selected_city}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/cities/category_pie_city_{selected_city.lower()}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of bombing categories for raids on {selected_city}*")
        else:
            st.info(f"No category distribution visualization available for {selected_city}")
        
        # Check for Schweinfurt-specific analysis
        if selected_city.upper() == "SCHWEINFURT":
            st.subheader("Special Analysis: Schweinfurt Ball Bearing Plant Raids")
            
            schweinfurt_path = f"{PLOT_PATH}/cities/schweinfurt/raids_timeline.png"
            if os.path.exists(schweinfurt_path):
                img = load_image(schweinfurt_path)
                if img:
                    st.image(img, use_container_width=True)
                    st.markdown("*Timeline of Schweinfurt raids showing tonnage and area bombing scores*")
            
            comparison_path = f"{PLOT_PATH}/cities/schweinfurt/schweinfurt_vs_other_bearings.png"
            if os.path.exists(comparison_path):
                img = load_image(comparison_path)
                if img:
                    st.image(img, use_container_width=True)
                    st.markdown("*Comparison of Schweinfurt to other ball bearing factory targets*")
        
        # Add city-specific data table
        st.markdown("---")
        st.subheader(f"Raid Data for {selected_city}")
        city_df = get_filtered_df(df, "city", selected_city)
        display_data_table(city_df)

elif page == "Category Analysis":
    st.title("Target Category Analysis")
    
    # Category selection
    selected_category = st.selectbox("Select a target category:", categories)
    
    if selected_category:
        st.markdown(f"## Bombing Analysis for {selected_category} Targets")
        
        # Display category-specific visualizations stacked vertically
        # Score distribution
        score_dist_path = f"{PLOT_PATH}/categories/score_distribution_category_{selected_category.lower()}.png"
        if os.path.exists(score_dist_path):
            img = load_image(score_dist_path)
            if img:
                st.subheader("Area Bombing Score Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of area bombing scores for {selected_category} targets*")
        else:
            st.info(f"No score distribution visualization available for {selected_category}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/categories/tonnage_vs_incendiary_category_{selected_category.lower()}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img, use_container_width=True)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage for {selected_category} targets*")
        else:
            st.info(f"No tonnage visualization available for {selected_category}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/categories/scores_by_target_type_category_{selected_category.lower()}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img, use_container_width=True)
                st.markdown(f"*Breakdown of area bombing scores by target type within {selected_category} category*")
        else:
            st.info(f"No target type visualization available for {selected_category}")
        
        # Component radar
        radar_path = f"{PLOT_PATH}/categories/component_radar_category_{selected_category.lower()}.png"
        if os.path.exists(radar_path):
            img = load_image(radar_path)
            if img:
                st.subheader("Component Score Analysis")
                st.image(img, use_container_width=True)
                st.markdown(f"*Radar chart showing component scores for {selected_category} targets*")
        else:
            st.info(f"No component analysis visualization available for {selected_category}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/categories/category_pie_category_{selected_category.lower()}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of bombing categories for {selected_category} targets*")
        else:
            st.info(f"No category distribution visualization available for {selected_category}")
        
        # Add category-specific data table
        st.markdown("---")
        st.subheader(f"Raid Data for {selected_category} Targets")
        category_df = get_filtered_df(df, "category", selected_category)
        display_data_table(category_df)

elif page == "Year Analysis":
    st.title("Yearly Bombing Analysis")
    
    # Year selection
    selected_year = st.selectbox("Select a year:", years)
    
    if selected_year:
        st.markdown(f"## Bombing Analysis for {selected_year}")
        
        # Display year-specific visualizations stacked vertically
        # Score distribution
        score_dist_path = f"{PLOT_PATH}/years/score_distribution_year_{selected_year}.png"
        if os.path.exists(score_dist_path):
            img = load_image(score_dist_path)
            if img:
                st.subheader("Area Bombing Score Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of area bombing scores during {selected_year}*")
        else:
            st.info(f"No score distribution visualization available for {selected_year}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/years/tonnage_vs_incendiary_year_{selected_year}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img, use_container_width=True)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage in {selected_year}*")
        else:
            st.info(f"No tonnage visualization available for {selected_year}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/years/scores_by_target_type_year_{selected_year}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img, use_container_width=True)
                st.markdown(f"*Breakdown of area bombing scores by target type in {selected_year}*")
        else:
            st.info(f"No target type visualization available for {selected_year}")
        
        # Component radar
        radar_path = f"{PLOT_PATH}/years/component_radar_year_{selected_year}.png"
        if os.path.exists(radar_path):
            img = load_image(radar_path)
            if img:
                st.subheader("Component Score Analysis")
                st.image(img, use_container_width=True)
                st.markdown(f"*Radar chart showing component scores for raids in {selected_year}*")
        else:
            st.info(f"No component analysis visualization available for {selected_year}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/years/category_pie_year_{selected_year}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img, use_container_width=True)
                st.markdown(f"*Distribution of bombing categories during {selected_year}*")
        else:
            st.info(f"No category distribution visualization available for {selected_year}")
        
        # Add year-specific data table
        st.markdown("---")
        st.subheader(f"Raid Data for {selected_year}")
        year_df = get_filtered_df(df, "year", selected_year)
        display_data_table(year_df)

elif page == "Data Download":
    st.title("Download Raw Data")
    
    st.markdown("""
    ## Available Datasets
    
    This section allows you to download the raw data used in this analysis.
    """)
    
    # USAAF Raids Data
    st.subheader("USAAF Raids Data")
    st.markdown("""
    Processed USAAF bombing raid data with area bombing classifications
    and metrics used in this analysis.
    """)
    download_link = get_download_link(PROCESSED_DATA_PATH, "Download USAAF Raids Data (CSV)")
    if download_link:
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("Unable to create download link for USAAF raids data")
    
    # Original Combined Attack Data
    st.subheader("Original Combined Attack Data")
    st.markdown("""
    Original combined bombing data before processing and filtering for USAAF raids.
    Contains raw data extracted from historical bombing records.
    """)
    download_link = get_download_link(ORIGINAL_DATA_PATH, "Download Original Data (CSV)")
    if download_link:
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("Unable to create download link for original data")
    
    st.markdown("---")
    
    # Add data preview
    st.subheader("Data Preview")
    tabs = st.tabs(["USAAF Raids Data", "Original Combined Data"])
    
    with tabs[0]:
        try:
            raids_data = pd.read_csv(PROCESSED_DATA_PATH)
            st.dataframe(raids_data.head(20), use_container_width=True)
            st.markdown(f"*Showing preview of {PROCESSED_DATA_PATH} ({len(raids_data)} records)*")
        except Exception as e:
            st.error(f"Error loading USAAF raids data: {e}")
    
    with tabs[1]:
        try:
            original_data = pd.read_csv(ORIGINAL_DATA_PATH)
            st.dataframe(original_data.head(20), use_container_width=True)
            st.markdown(f"*Showing preview of {ORIGINAL_DATA_PATH} ({len(original_data)} records)*")
        except Exception as e:
            st.error(f"Error loading original combined data: {e}")
    
    st.markdown("---")
    
    st.markdown("""
    ## Data Processing Information
    
    The visualizations in this dashboard are based on pre-processed datasets derived from
    historical bombing records. The processing workflow included:
    
    1. Data extraction from original bombing records
    2. Cleaning and standardization of location names, dates, and metrics
    3. Calculation of area bombing scores based on target type, tonnage, and incendiary percentages
    4. Classification of raids into bombing categories
    5. Generation of visualizations to analyze patterns and trends
    
    For questions about the data processing methodology or to report issues,
    please contact the repository maintainer.
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
USAAF Bombing Campaign Analysis Dashboard | World War II Bombing Data
</div>
""", unsafe_allow_html=True) 