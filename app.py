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
    ## Challenging the Conventional Narrative
    
    Images of Dresden reduced to rubble and Hamburg engulfed in flames have become emblematic of Allied bombing campaigns. The conventional narrative suggests that the United States gradually abandoned its commitment to precision bombing in favor of devastating area attacks as the war intensified.
    
    **This dataset and analysis challenges that narrative.** Based on the first comprehensive digitization of every bombing raid in the European theater, this research reveals no evidence of the commonly asserted wholesale shift from precision to area bombing. While area bombing did increase modestly in later years and certain symbolic targets received disproportionately heavy treatment, precision methods persistently dominated throughout the conflict.
    
    Using a novel classification system that considers target designation, incendiary usage, and tonnage deployed, this analysis provides a data-driven perspective on the strategic bombing campaign.
    """)
    
    # Display general visualizations stacked vertically
    st.subheader("Overall Distribution of Area Bombing Scores")
    img = load_image(f"{PLOT_PATH}/general/overall_score_distribution.png")
    if img:
        st.image(img)
        st.markdown("*The mean area bombing score across all USAAF raids was 3.24 out of 10, with a median of 2.6—indicating that most bombing operations maintained significant precision elements throughout the war with a long-tail of outlier area bombing raids.*")
    
    st.subheader("Tonnage Distribution")
    img = load_image(f"{PLOT_PATH}/general/tonnage_distribution.png")
    if img:
        st.image(img)
        st.markdown("*Distribution of bombing tonnage per raid across all USAAF missions. While tonnage increased over time, this represented a quantitative rather than qualitative evolution in bombing operations.*")
    
    st.subheader("HE vs Incendiary Bombing by Year")
    img = load_image(f"{PLOT_PATH}/general/he_vs_incendiary_by_year.png")
    if img:
        st.image(img)
        st.markdown("*Annual comparison of high explosive vs. incendiary bombing tonnage. Notably, the incendiary percentage remained relatively flat throughout, contradicting the notion of a progressive shift toward fire-based area bombing tactics.*")
    
    st.subheader("Relationship Between Tonnage and Area Bombing")
    img = load_image(f"{PLOT_PATH}/general/tonnage_vs_score_relationship.png")
    if img:
        st.image(img)
        st.markdown("*Correlation between bombing tonnage and area bombing scores. The data shows that target type, not tonnage, was the primary determinant of whether a mission employed area bombing characteristics.*")
    
    st.subheader("Bombing Patterns Over Time")
    img = load_image(f"{PLOT_PATH}/general/quarterly_metrics_evolution.png")
    if img:
        st.image(img)
        st.markdown("*Quarterly evolution of key bombing metrics throughout the war. While there was a modest upward trend in area bombing scores over time (from approximately 2.5 to 3.5), this change falls well within one standard deviation and represents a refinement rather than transformation of bombing doctrine.*")
        
    img = load_image(f"{PLOT_PATH}/general/tonnage_distribution_by_category.png")
    if img:
        st.image(img)
        st.markdown("*Distribution of bombing tonnage across different target categories. Transportation, oil refineries, and airfields received the most tonnage, reflecting strategic priorities.*")
    
    st.subheader("Temporal Analysis by Target Category")
    img = load_image(f"{PLOT_PATH}/general/year_category_score_heatmap.png")
    if img:
        st.image(img)
        st.markdown("*Evolution of area bombing scores by target category and year. No category demonstrated the dramatic escalation that would indicate either a deliberate concealment of area tactics or a fundamental doctrinal shift.*")
        
    st.subheader("Monthly Progression")
    img = load_image(f"{PLOT_PATH}/general/monthly_score_progression.png")
    if img:
        st.image(img)
        st.markdown("*Monthly progression of area bombing scores with marker size representing tonnage. The modest increase in area bombing coincided with the period of heaviest overall bombing activity, suggesting operational scale rather than doctrinal transformation drove these changes.*")
    
    # Add the category by year visualization
    st.subheader("Distribution of Bombing Categories by Year")
    img = load_image(f"{PLOT_PATH}/years/category_by_year.png")
    if img:
        st.image(img)
        st.markdown("*Evolution of bombing categories throughout the war years. Very precise bombing (scores 0-2) maintained a relatively stable proportion throughout most of the war. Mixed bombing approaches (scores 4-6) increased modestly from around 10% early in the war to 18.6% at their peak.*")
    
    # Add complete dataset view
    st.markdown("---")
    st.subheader("Explore Complete Dataset")
    st.markdown("""
    This dashboard presents the first comprehensive digitization of the United States Strategic Bombing Survey (USSBS) attack-by-attack data, 
    involving over 8,000 early computer readouts from the National Archives. Explore the data below to examine patterns across the entire campaign.
    """)
    display_data_table(df)

elif page == "City Analysis":
    st.title("City-Specific Bombing Analysis")
    st.markdown("""
    The city-specific analysis reveals important variations in bombing patterns. While most cities maintained predominantly precision bombing characteristics,
    certain symbolic targets like Berlin received significantly different treatment.
    
    Berlin stands as an exception to overall patterns, with only 23% of operations qualifying as very precise or precise bombing, 
    and nearly 27% constituting clear or heavy area bombing. The average incendiary percentage for Berlin proper reached 46%—extraordinarily high compared to other targets.
    This exceptional treatment of the Nazi capital suggests that emotional or symbolic factors influenced targeting decisions for particularly emblematic objectives.
    """)
    
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
                st.image(img)
                st.markdown(f"*Distribution of area bombing scores for raids targeting {selected_city}. The distribution reveals the extent to which operations against this city adhered to or deviated from precision bombing approaches.*")
        else:
            st.info(f"No score distribution visualization available for {selected_city}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/cities/tonnage_vs_incendiary_city_{selected_city.lower()}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage for {selected_city} raids. Incendiary percentage is a key indicator of area bombing tactics, with higher percentages generally associated with urban area targeting rather than precision strikes.*")
        else:
            st.info(f"No tonnage visualization available for {selected_city}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/cities/scores_by_target_type_city_{selected_city.lower()}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img)
                st.markdown(f"*Breakdown of area bombing scores by target type within {selected_city}. Different target types within the same city often received distinctly different bombing approaches.*")
        else:
            st.info(f"No target type visualization available for {selected_city}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/cities/category_pie_city_{selected_city.lower()}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img)
                st.markdown(f"*Distribution of bombing categories for raids on {selected_city}. This chart shows the proportion of raids falling into each bombing category from very precise (0-2) to heavy area (8-10).*")
        else:
            st.info(f"No category distribution visualization available for {selected_city}")
        
        # Check for Schweinfurt-specific analysis
        if selected_city.upper() == "SCHWEINFURT":
            st.subheader("Special Analysis: Schweinfurt Ball Bearing Plant Raids")
            
            schweinfurt_path = f"{PLOT_PATH}/cities/schweinfurt/raids_timeline.png"
            if os.path.exists(schweinfurt_path):
                img = load_image(schweinfurt_path)
                if img:
                    st.image(img)
                    st.markdown("*Timeline of Schweinfurt raids showing tonnage and area bombing scores. The Schweinfurt raids represent an important case study in precision bombing efforts against a critical industrial target.*")
            
            comparison_path = f"{PLOT_PATH}/cities/schweinfurt/schweinfurt_vs_other_bearings.png"
            if os.path.exists(comparison_path):
                img = load_image(comparison_path)
                if img:
                    st.image(img)
                    st.markdown("*Comparison of Schweinfurt to other ball bearing factory targets. Despite their strategic importance, Schweinfurt raids maintained precision characteristics rather than shifting to area bombing tactics.*")
        
        # Add city-specific data table
        st.markdown("---")
        st.subheader(f"Raid Data for {selected_city}")
        city_df = get_filtered_df(df, "city", selected_city)
        display_data_table(city_df)

elif page == "Category Analysis":
    st.title("Target Category Analysis")
    st.markdown("""
    Breaking down the data by target category provides compelling evidence against a hidden shift toward area bombing. 
    
    One persistent claim in strategic bombing historiography deserves particular scrutiny: the assertion that transportation targets served as a pretext for area bombing.
    Several prominent scholars have advanced this theory, arguing that transportation targeting masked deliberate attacks on civilian populations.
    
    Our comprehensive data analysis thoroughly refutes these claims. Transportation targets consistently maintained one of the most precise bombing profiles throughout the war, 
    with a median area bombing score of just 2.4. The incendiary component averaged a mere 1.0% across all transportation raids, significantly lower than the dataset average.
    
    Examine different target categories below to see how bombing characteristics varied by target type.
    """)
    
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
                st.image(img)
                st.markdown(f"*Distribution of area bombing scores for {selected_category} targets. This distribution reveals the extent to which operations against this target category adhered to precision or area bombing approaches.*")
        else:
            st.info(f"No score distribution visualization available for {selected_category}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/categories/tonnage_vs_incendiary_category_{selected_category.lower()}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage for {selected_category} targets. The pattern of incendiary usage provides insight into tactical approaches for this category.*")
        else:
            st.info(f"No tonnage visualization available for {selected_category}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/categories/scores_by_target_type_category_{selected_category.lower()}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img)
                st.markdown(f"*Breakdown of area bombing scores by target type within {selected_category} category. Different target sub-types often received distinctly different bombing approaches.*")
        else:
            st.info(f"No target type visualization available for {selected_category}")
        
        # Component radar
        radar_path = f"{PLOT_PATH}/categories/component_radar_category_{selected_category.lower()}.png"
        if os.path.exists(radar_path):
            img = load_image(radar_path)
            if img:
                st.subheader("Component Score Analysis")
                st.image(img)
                st.markdown(f"*Radar chart showing component scores for {selected_category} targets. This visualization breaks down how target type, incendiary percentage, and tonnage each contributed to the overall area bombing score.*")
        else:
            st.info(f"No component analysis visualization available for {selected_category}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/categories/category_pie_category_{selected_category.lower()}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img)
                st.markdown(f"*Distribution of bombing categories for {selected_category} targets. This chart shows the proportion of raids falling into each bombing category from very precise (0-2) to heavy area (8-10).*")
        else:
            st.info(f"No category distribution visualization available for {selected_category}")
        
        # Add category-specific data table
        st.markdown("---")
        st.subheader(f"Raid Data for {selected_category} Targets")
        category_df = get_filtered_df(df, "category", selected_category)
        display_data_table(category_df)

elif page == "Year Analysis":
    st.title("Yearly Bombing Analysis")
    st.markdown("""
    The conventional narrative suggests an inexorable progression from precision to area bombing as the war intensified. 
    However, year-by-year analysis reveals a much more nuanced reality.
    
    Rather than a dramatic shift, we find relative stability throughout the conflict. While there was a modest upward trend 
    in area bombing scores over time (from approximately 2.5 to 3.5), this change falls well within one standard deviation 
    and represents a refinement rather than transformation of bombing doctrine.
    
    The yearly distribution of bombing categories shows that very precise bombing (scores 0-2) decreased slightly from 1940 to 1941 
    but then maintained a relatively stable proportion throughout the remainder of the war. 
    Mixed bombing approaches (scores 4-6) increased modestly from around 10% early in the war to 18.6% at their peak.
    
    Select a year below to examine its specific bombing patterns.
    """)
    
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
                st.image(img)
                st.markdown(f"*Distribution of area bombing scores during {selected_year}. This distribution reveals how bombing missions during this year fell along the precision-to-area spectrum.*")
        else:
            st.info(f"No score distribution visualization available for {selected_year}")
        
        # Tonnage vs incendiary
        tonnage_path = f"{PLOT_PATH}/years/tonnage_vs_incendiary_year_{selected_year}.png"
        if os.path.exists(tonnage_path):
            img = load_image(tonnage_path)
            if img:
                st.subheader("Tonnage vs Incendiary Percentage")
                st.image(img)
                st.markdown(f"*Relationship between bombing tonnage and incendiary percentage in {selected_year}. The pattern reveals tactical approaches employed during this period of the war.*")
        else:
            st.info(f"No tonnage visualization available for {selected_year}")
        
        # Target type analysis
        target_path = f"{PLOT_PATH}/years/scores_by_target_type_year_{selected_year}.png"
        if os.path.exists(target_path):
            img = load_image(target_path)
            if img:
                st.subheader("Scores by Target Type")
                st.image(img)
                st.markdown(f"*Breakdown of area bombing scores by target type in {selected_year}. This visualization reveals how different target types were approached during this year.*")
        else:
            st.info(f"No target type visualization available for {selected_year}")
        
        # Component radar
        radar_path = f"{PLOT_PATH}/years/component_radar_year_{selected_year}.png"
        if os.path.exists(radar_path):
            img = load_image(radar_path)
            if img:
                st.subheader("Component Score Analysis")
                st.image(img)
                st.markdown(f"*Radar chart showing component scores for raids in {selected_year}. This breaks down how target type, incendiary percentage, and tonnage each contributed to the overall area bombing score during this period.*")
        else:
            st.info(f"No component analysis visualization available for {selected_year}")
        
        # Category distribution
        category_path = f"{PLOT_PATH}/years/category_pie_year_{selected_year}.png"
        if os.path.exists(category_path):
            img = load_image(category_path)
            if img:
                st.subheader("Bombing Category Distribution")
                st.image(img)
                st.markdown(f"*Distribution of bombing categories during {selected_year}. This chart shows the proportion of raids falling into each bombing category from very precise (0-2) to heavy area (8-10).*")
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
    ## Research Methodology & Data Access
    
    This dashboard presents the first comprehensive digitization of the United States Strategic Bombing Survey (USSBS) attack-by-attack data.
    This unprecedented effort involved processing over 8,000 early computer readouts from the National Archives, 
    extracting locations, tonnage, munition types, and other critical operational details for every recorded bombing mission in the European theater.
    
    Prior historical analyses have typically focused on specific campaigns or cities without examining the complete dataset, 
    leaving scholars unable to identify broader patterns or make definitive claims about the overall character of the air war.
    
    To systematically evaluate the nature of each bombing mission, we developed a three-dimensional scoring algorithm that considers:
    
    1. **Target designation**: Whether the USSBS categorized the target as a city area attack
    2. **Incendiary proportion**: The percentage of incendiary munitions used relative to total tonnage
    3. **Total tonnage**: Whether the mission employed excessive tonnage compared to operational norms
    
    This multidimensional approach provides a nuanced assessment of bombing character beyond the binary precision/area classification 
    often employed in historical narratives.
    
    The complete datasets are available for download below.
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
    ## Research Findings
    
    The persistence of precision bombing throughout the war fundamentally challenges the traditional historical narrative. 
    The conventional explanation—that a wholesale shift from precision to area bombing occurred—is not supported by the empirical evidence.
    
    Our comprehensive analysis of the raw data reveals a more nuanced reality. While area bombing did increase in specific instances like Berlin 
    and occurred more frequently in absolute terms as operations expanded, it remained proportionally consistent and concentrated 
    in particular symbolic targets rather than representing a doctrinal transformation.
    
    The campaign was predominantly characterized by precision approaches, with a mean area bombing score of just 3.24 out of 10 across all raids. 
    This finding directly contradicts the commonly held notion of a dramatic "0 to 1" shift from precision to area bombing by the USAAF throughout the war.
    
    Instead, the evidence points to a strategic continuity with tactical adaptations, maintaining precision as the fundamental operational approach 
    even as the scale and complexity of the air campaign reached unprecedented levels.
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
USAAF Bombing Campaign Analysis Dashboard | The first comprehensive digitization of every WWII bombing raid in the European theater
</div>
""", unsafe_allow_html=True) 