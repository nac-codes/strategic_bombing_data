import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300

# Create directories for plots
for dir_name in ['plots/usaaf/years', 'plots/usaaf/categories', 'plots/usaaf/cities', 'plots/usaaf/general']:
    os.makedirs(dir_name, exist_ok=True)

# Load and merge data - using USAAF filtered data
print("Loading USAAF-only data...")
raids_classification = pd.read_csv('processed_data/usaaf/usaaf_raids_classification_with_category.csv')
raids_summary = pd.read_csv('processed_data/raids_summary.csv')

# Display column names to debug
print("Raids classification columns:", raids_classification.columns.tolist())
print("Raids summary columns:", raids_summary.columns.tolist())

# Create a unique identifier to merge datasets
raids_classification['raid_id'] = raids_classification.index
raids_summary['raid_id'] = raids_summary.index

# Merge datasets
print("Merging datasets...")
# Use the dataset that already has all needed columns
df = raids_classification.copy()

# Clean up year data - it should be 1940s
print("Cleaning year data...")
df['YEAR'] = df['YEAR'].fillna(0).astype(float)  # Handle missing values
df['Year'] = (1940 + df['YEAR']).astype(int)
# Handle any outlier years
df.loc[df['Year'] < 1939, 'Year'] = 1940
df.loc[df['Year'] > 1946, 'Year'] = 1945

# Create a clean location field
df['Location'] = df['target_location'].str.strip().str.upper()

# Identify top 10 cities
top_cities = df['Location'].value_counts().head(10).index.tolist()
print(f"Top 10 most raided cities by USAAF: {', '.join(top_cities)}")

# Add Schweinfurt to the analysis if not already in top cities
if 'SCHWEINFURT' not in top_cities:
    top_cities.append('SCHWEINFURT')
    print("Added Schweinfurt to city analysis list")

# Set up a common color map for consistency
cmap = plt.cm.viridis
norm = plt.Normalize(0, 10)

# Create a custom function to generate standard set of plots
def generate_plots(data, group_name, save_dir):
    """Generate standard set of plots for a given dataset subset"""
    
    # 1. Distribution of Area Bombing Scores
    plt.figure(figsize=(12, 6))
    sns.histplot(data['AREA_BOMBING_SCORE_NORMALIZED'], bins=20, kde=True)
    plt.title(f'Distribution of Area Bombing Scores - {group_name} (USAAF)', fontsize=16)
    plt.xlabel('Area Bombing Score (10 = Clear Area Bombing, 0 = Precise Bombing)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.axvline(data['AREA_BOMBING_SCORE_NORMALIZED'].median(), color='red', linestyle='--', 
               label=f'Median: {data["AREA_BOMBING_SCORE_NORMALIZED"].median():.1f}')
    plt.xlim(0, 10)  # Enforce consistent x-axis limits
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/score_distribution_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()
    
    # 2. Scatter plot of Tonnage vs Incendiary Score
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        data['TOTAL_TONS'].clip(0, 500), 
        data['INCENDIARY_PERCENT'].clip(0, 100), 
        c=data['AREA_BOMBING_SCORE_NORMALIZED'], 
        alpha=0.7, 
        cmap='viridis', 
        s=40,
        norm=norm
    )
    plt.colorbar(scatter, label='Area Bombing Score')
    plt.title(f'Tonnage vs Incendiary Percentage - {group_name} (USAAF)', fontsize=16)
    plt.xlabel('Total Tons (clipped at 500)', fontsize=12)
    plt.ylabel('Incendiary Percentage', fontsize=12)
    plt.xlim(0, 500)  # Enforce consistent x-axis limits
    plt.ylim(0, 100)  # Enforce consistent y-axis limits
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tonnage_vs_incendiary_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()
    
    # 3. Box plot of scores by target type
    plt.figure(figsize=(10, 6))
    data.loc[:, 'Target Type'] = data['TARGET_SCORE'].map({1: 'Industrial/Area', 0: 'Non-Industrial/Precision'})
    sns.boxplot(x='Target Type', y='AREA_BOMBING_SCORE_NORMALIZED', data=data)
    plt.title(f'Area Bombing Scores by Target Type - {group_name} (USAAF)', fontsize=16)
    plt.xlabel('Target Type', fontsize=12)
    plt.ylabel('Area Bombing Score', fontsize=12)
    plt.ylim(0, 10)  # Enforce consistent y-axis limits
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scores_by_target_type_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()
    
    # 4. Category distribution (pie chart)
    if len(data) > 20:  # Only create if enough data points
        plt.figure(figsize=(12, 10))
        cat_counts = data['Score Category'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cat_counts)))
        cat_counts.plot.pie(autopct='%1.1f%%', colors=colors, explode=[0.05]*len(cat_counts),
                           textprops={'fontsize': 12})
        plt.title(f'Distribution of Bombing Categories - {group_name} (USAAF)', fontsize=16)
        plt.ylabel('')  # Hide the ylabel
        plt.tight_layout()
        plt.savefig(f'{save_dir}/category_pie_{group_name.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    # 5. Component scores (radar chart for groups with sufficient data)
    if len(data) >= 5:
        # Calculate average component scores
        avg_target = data['TARGET_SCORE'].mean() * 10
        avg_tonnage = data['TONNAGE_SCORE'].mean()
        avg_incendiary = data['INCENDIARY_SCORE'].mean()
        
        # Create radar chart
        categories = ['Target Type', 'Tonnage', 'Incendiary']
        values = [avg_target, avg_tonnage, avg_incendiary]
        
        # Create the radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color='darkblue')
        ax.fill(angles, values, alpha=0.25, color='darkblue')
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 10)  # Consistent range for component scores
        ax.set_title(f'Average Component Scores - {group_name} (USAAF)', fontsize=16, pad=20)
        ax.grid(True)
        
        # Add values at points
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.5, f'{value:.1f}', 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/component_radar_{group_name.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()

# Add score category to data
print("Creating score categories...")
df['Score Category'] = pd.cut(df['AREA_BOMBING_SCORE_NORMALIZED'], 
                             bins=[0, 2, 4, 6, 8, 10],
                             labels=['Very Precise (0-2)', 'Precise (2-4)', 
                                    'Mixed (4-6)', 'Area (6-8)', 'Heavy Area (8-10)'])

print("Generating plots by year...")
# 1. Generate plots by year
years = sorted(df['Year'].unique())
for year in years:
    if pd.notna(year) and year >= 1940 and year <= 1945:
        year_data = df[df['Year'] == year]
        if len(year_data) > 0:
            print(f"  Processing year {year} ({len(year_data)} raids)")
            generate_plots(year_data, f"Year {year}", "plots/usaaf/years")

# Year-specific visualizations
print("Creating year evolution plots...")
# 1. Evolution of bombing scores over years
plt.figure(figsize=(14, 8))
yearly_scores = df.groupby('Year')['AREA_BOMBING_SCORE_NORMALIZED'].agg(['mean', 'median', 'std']).reset_index()
yearly_scores = yearly_scores[(yearly_scores['Year'] >= 1940) & (yearly_scores['Year'] <= 1945)]

plt.errorbar(yearly_scores['Year'], yearly_scores['mean'], yerr=yearly_scores['std'], 
             fmt='o-', capsize=5, capthick=2, label='Mean Score ± Std Dev')
plt.plot(yearly_scores['Year'], yearly_scores['median'], 's--', color='red', label='Median Score')
plt.grid(True, alpha=0.3)
plt.title('Evolution of Area Bombing Scores Throughout the War (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Area Bombing Score', fontsize=14)
plt.xticks(yearly_scores['Year'], fontsize=12)
plt.ylim(0, 10)  # Consistent y-axis limit
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/usaaf/years/yearly_evolution.png', dpi=300)
plt.close()

# 2. Stacked bar chart of bombing categories by year
plt.figure(figsize=(14, 8))
year_cat = pd.crosstab(df['Year'], df['Score Category'])
year_cat = year_cat.loc[year_cat.index.astype(float).astype(int).isin(range(1940, 1946))]
year_cat_pct = year_cat.div(year_cat.sum(axis=1), axis=0) * 100

year_cat_pct.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution of Bombing Categories by Year (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Raids', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.ylim(0, 100)  # Consistent y-axis limit for percentage plots
plt.legend(title='Bombing Category', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3, axis='y')

# Add percentage text on bars
for i, year in enumerate(year_cat_pct.index):
    cumulative_sum = 0
    for j, col in enumerate(year_cat_pct.columns):
        # Only add text for segments that are at least 5% of the total
        if year_cat_pct.loc[year, col] >= 5:
            plt.text(i, cumulative_sum + year_cat_pct.loc[year, col]/2,
                    f"{year_cat_pct.loc[year, col]:.1f}%", ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        cumulative_sum += year_cat_pct.loc[year, col]

plt.tight_layout()
plt.savefig('plots/usaaf/years/category_by_year.png', dpi=300)
plt.close()

print("Generating plots by category...")
# 2. Generate plots by category
for category in df['CATEGORY'].unique():
    if pd.notna(category):
        category_data = df[df['CATEGORY'] == category]
        print(f"  Processing category {category} ({len(category_data)} raids)")
        generate_plots(category_data, f"Category {category}", "plots/usaaf/categories")

# Category-specific visualizations
print("Creating category comparison plots...")
# 1. Box plot comparing categories
plt.figure(figsize=(16, 10))
# Sort categories by median score
cat_medians = df.groupby('CATEGORY')['AREA_BOMBING_SCORE_NORMALIZED'].median().sort_values(ascending=False)
top_categories = cat_medians.index[:12]  # Focus on top 12 categories for readability

cat_subset = df[df['CATEGORY'].isin(top_categories)]
ax = sns.boxplot(x='CATEGORY', y='AREA_BOMBING_SCORE_NORMALIZED', 
                order=cat_medians.index[:12],
                data=cat_subset, palette='viridis')
plt.title('Area Bombing Scores by Target Category (USAAF)', fontsize=18)
plt.xlabel('Target Category', fontsize=14)
plt.ylabel('Area Bombing Score', fontsize=14)
plt.ylim(0, 10)  # Consistent y-axis limit
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/usaaf/categories/category_comparison.png', dpi=300)
plt.close()

# 2. Heatmap of category and year
plt.figure(figsize=(16, 10))
pivot_data = df.pivot_table(
    values='AREA_BOMBING_SCORE_NORMALIZED',
    index='CATEGORY',
    columns='Year',
    aggfunc='mean'
)
# Focus on years 1940-1945 and categories with sufficient data
pivot_filtered = pivot_data.loc[pivot_data.count(axis=1) >= 3, [y for y in range(1940, 1946) if y in pivot_data.columns]]
pivot_filtered = pivot_filtered.dropna(thresh=3)  # Drop rows with too many NaNs

# Set consistent color scale for heatmap
sns.heatmap(pivot_filtered, cmap='viridis', annot=True, fmt='.1f', linewidths=0.5, 
            cbar_kws={'label': 'Average Area Bombing Score'}, vmin=0, vmax=10)
plt.title('Average Area Bombing Score by Category and Year (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Target Category', fontsize=14)
plt.tight_layout()
plt.savefig('plots/usaaf/categories/category_year_heatmap.png', dpi=300)
plt.close()

print("Generating plots by city...")
# 3. Generate plots for top 50 cities by number of raids
city_raid_counts = df['Location'].value_counts()
top_50_cities = city_raid_counts.nlargest(50).index.tolist()

for city in top_50_cities:
    if pd.notna(city) and city.strip():  # Skip empty or NaN values
        city_data = df[df['Location'] == city]
        print(f"  Processing city {city} ({len(city_data)} raids)")
        generate_plots(city_data, f"City {city}", "plots/usaaf/cities")

# City-specific visualizations
print("Creating city comparison plots...")
# 1. Bar chart of average bombing scores for top cities
plt.figure(figsize=(14, 8))
city_scores = df[df['Location'].isin(top_cities)].groupby('Location')['AREA_BOMBING_SCORE_NORMALIZED'].mean().sort_values(ascending=False)

# Create a colormap based on the scores
colors = plt.cm.viridis(np.linspace(0, 1, len(city_scores)))

city_scores.plot(kind='bar', color=colors)
plt.title('Average Area Bombing Score by City (USAAF)', fontsize=18)
plt.xlabel('City', fontsize=14)
plt.ylabel('Average Area Bombing Score', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 10)  # Consistent y-axis limit

# Add values above bars
for i, v in enumerate(city_scores):
    plt.text(i, v + 0.2, f"{v:.1f}", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/usaaf/cities/city_comparison.png', dpi=300)
plt.close()

# 2. Evolution of bombing intensity for top 5 cities
plt.figure(figsize=(14, 8))
top5_cities = city_scores.index[:5]  # Top 5 most heavily bombed cities by score
city_year_counts = df[df['Location'].isin(top5_cities)].pivot_table(
    values='AREA_BOMBING_SCORE_NORMALIZED', 
    index='Year', 
    columns='Location', 
    aggfunc='mean'
)

# Filter to just 1940-1945
city_year_counts = city_year_counts.loc[city_year_counts.index.astype(int).isin(range(1940, 1946))]

for city in city_year_counts.columns:
    plt.plot(city_year_counts.index, city_year_counts[city], 'o-', linewidth=2, markersize=8, label=city)

plt.title('Evolution of Bombing Strategy for Top 5 Cities (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Area Bombing Score', fontsize=14)
plt.xticks(city_year_counts.index, fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 10)  # Consistent y-axis limit
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/usaaf/cities/city_evolution.png', dpi=300)
plt.close()

# 3. Stacked bar chart showing proportion of raid categories for top cities
plt.figure(figsize=(14, 8))
city_categories = pd.crosstab(df[df['Location'].isin(top_cities)]['Location'], df[df['Location'].isin(top_cities)]['Score Category'])
city_categories_pct = city_categories.div(city_categories.sum(axis=1), axis=0) * 100

# Sort by proportion of area bombing (highest to lowest)
area_proportion = city_categories_pct['Area (6-8)'] + city_categories_pct['Heavy Area (8-10)']
sorted_cities = area_proportion.sort_values(ascending=False).index

city_categories_pct = city_categories_pct.loc[sorted_cities]

city_categories_pct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
plt.title('Distribution of Bombing Categories by City (USAAF)', fontsize=18)
plt.xlabel('City', fontsize=14)
plt.ylabel('Percentage of Raids', fontsize=14)
plt.ylim(0, 100)  # Consistent y-axis limit for percentage plots
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Bombing Category', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3, axis='y')

# Add percentage text on bars
for i, city in enumerate(city_categories_pct.index):
    cumulative_sum = 0
    for j, col in enumerate(city_categories_pct.columns):
        # Only add text for segments that are at least 5% of the total
        if city_categories_pct.loc[city, col] >= 5:
            plt.text(i, cumulative_sum + city_categories_pct.loc[city, col]/2,
                    f"{city_categories_pct.loc[city, col]:.1f}%", ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        cumulative_sum += city_categories_pct.loc[city, col]

plt.tight_layout()
plt.savefig('plots/usaaf/cities/category_by_city.png', dpi=300)
plt.close()

# 4. Generate overall comparison with RAF (combined dataset)
print("Creating USAAF/RAF comparison plots...")
# Load full dataset
all_raids = pd.read_csv('processed_data/raids_area_bombing_classification.csv')
all_raids['raid_id'] = all_raids.index

# Try to merge with raids_summary to get AIR FORCE data if available
try:
    all_raids_with_af = pd.merge(
        all_raids,
        raids_summary[['raid_id', 'AIR FORCE']],
        on='raid_id',
        how='left'
    )
    
    # Create Air Force type
    all_raids_with_af['Air Force'] = all_raids_with_af['AIR FORCE'].apply(lambda x: 'RAF' if x == 'R' else 'USAAF')
    
    # Score distribution comparison
    plt.figure(figsize=(14, 8))
    sns.histplot(data=all_raids_with_af, x='AREA_BOMBING_SCORE_NORMALIZED', hue='Air Force', 
                element='step', stat='density', common_norm=False, bins=20, kde=True)
    plt.title('Area Bombing Score Distribution: USAAF vs RAF', fontsize=18)
    plt.xlabel('Area Bombing Score (10 = Clear Area Bombing, 0 = Precise Bombing)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig('plots/usaaf/usaaf_vs_raf_score_distribution.png', dpi=300)
    plt.close()
    
    # Category comparison
    plt.figure(figsize=(14, 8))
    all_raids_with_af['Score Category'] = pd.cut(all_raids_with_af['AREA_BOMBING_SCORE_NORMALIZED'], 
                                 bins=[0, 2, 4, 6, 8, 10],
                                 labels=['Very Precise (0-2)', 'Precise (2-4)', 
                                        'Mixed (4-6)', 'Area (6-8)', 'Heavy Area (8-10)'])
    
    category_by_af = pd.crosstab(all_raids_with_af['Air Force'], all_raids_with_af['Score Category'])
    category_by_af_pct = category_by_af.div(category_by_af.sum(axis=1), axis=0) * 100
    
    # Plot side by side
    category_by_af_pct.plot(kind='bar', figsize=(14, 8))
    plt.title('Bombing Categories: USAAF vs RAF', fontsize=18)
    plt.xlabel('Air Force', fontsize=14)
    plt.ylabel('Percentage of Raids', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage text on bars
    for i, af in enumerate(category_by_af_pct.index):
        cumulative_sum = 0
        for j, col in enumerate(category_by_af_pct.columns):
            # Only add text for segments that are at least 3% of the total
            if category_by_af_pct.loc[af, col] >= 3:
                plt.text(i, category_by_af_pct.loc[af, col]/2,
                        f"{category_by_af_pct.loc[af, col]:.1f}%", ha='center', va='center',
                        fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/usaaf/usaaf_vs_raf_categories.png', dpi=300)
    plt.close()
except Exception as e:
    print(f"Skipping RAF comparison due to error: {e}")

# Add new section for general bombing campaign visualizations
print("Creating general bombing campaign visualizations...")
os.makedirs('plots/usaaf/general', exist_ok=True)

# 1. Overall Distribution of Area Bombing Scores
plt.figure(figsize=(14, 8))
sns.histplot(df['AREA_BOMBING_SCORE_NORMALIZED'], bins=20, kde=True, color='darkblue')
plt.axvline(df['AREA_BOMBING_SCORE_NORMALIZED'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["AREA_BOMBING_SCORE_NORMALIZED"].mean():.2f}')
plt.axvline(df['AREA_BOMBING_SCORE_NORMALIZED'].median(), color='green', linestyle='-.',
            label=f'Median: {df["AREA_BOMBING_SCORE_NORMALIZED"].median():.2f}')
plt.title('Overall Distribution of Area Bombing Scores (USAAF)', fontsize=18)
plt.xlabel('Area Bombing Score (10 = Clear Area Bombing, 0 = Precise Bombing)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.tight_layout()
plt.savefig('plots/usaaf/general/overall_score_distribution.png', dpi=300)
plt.close()

# 2. Tonnage Distribution Analysis
plt.figure(figsize=(14, 8))
# Clip extreme values for better visualization
tonnage_data = df['TOTAL_TONS'].clip(0, 500)
sns.histplot(tonnage_data, bins=30, kde=True, color='darkgreen')
plt.axvline(tonnage_data.mean(), color='red', linestyle='--', 
            label=f'Mean: {tonnage_data.mean():.2f} tons')
plt.axvline(tonnage_data.median(), color='orange', linestyle='-.',
            label=f'Median: {tonnage_data.median():.2f} tons')
plt.title('Distribution of Bombing Tonnage (USAAF, clipped at 500 tons)', fontsize=18)
plt.xlabel('Total Tons Dropped', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/usaaf/general/tonnage_distribution.png', dpi=300)
plt.close()

# 3. Incendiary vs HE Bombing Analysis
plt.figure(figsize=(14, 8))
# Calculate HE tonnage (non-incendiary)
df['INCENDIARY_PERCENT'] = df['INCENDIARY_PERCENT'].fillna(0)  # Treat missing as 0% incendiary
df['HE_PERCENT'] = 100 - df['INCENDIARY_PERCENT']
df['HE_TONS'] = df['TOTAL_TONS'] * df['HE_PERCENT'] / 100
df['INCENDIARY_TONS'] = df['TOTAL_TONS'] * df['INCENDIARY_PERCENT'] / 100

# Group by year
bombing_by_year = df.groupby('Year').agg({
    'TOTAL_TONS': 'sum',
    'HE_TONS': 'sum', 
    'INCENDIARY_TONS': 'sum'
}).reset_index()

# Plot stacked bar chart
bombing_years = bombing_by_year[(bombing_by_year['Year'] >= 1940) & (bombing_by_year['Year'] <= 1945)]
bar_width = 0.8
plt.bar(bombing_years['Year'], bombing_years['HE_TONS'], 
        color='steelblue', width=bar_width, label='HE Bombs')
plt.bar(bombing_years['Year'], bombing_years['INCENDIARY_TONS'], 
        bottom=bombing_years['HE_TONS'], color='darkorange', width=bar_width, label='Incendiary Bombs')

plt.title('HE vs Incendiary Bombing by Year (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Tons', fontsize=14)
plt.xticks(bombing_years['Year'], fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add total tonnage labels
for i, year_data in enumerate(bombing_years.itertuples()):
    plt.text(year_data.Year, year_data.TOTAL_TONS + 1000, 
             f'{year_data.TOTAL_TONS:,.0f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/usaaf/general/he_vs_incendiary_by_year.png', dpi=300)
plt.close()

# 4. Monthly Progression of Bombing Scores
plt.figure(figsize=(16, 8))
# Create month-year field, handling NaN values
df['Month'] = df['MONTH'].fillna(1).astype(float).astype(int)  # Convert to float first then int
df['Month-Year'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)

# Filter to only include dates within the war period with sufficient data
monthly_data = df.groupby('Month-Year').agg({
    'AREA_BOMBING_SCORE_NORMALIZED': ['mean', 'median', 'count'],
    'TOTAL_TONS': 'sum'
}).reset_index()
monthly_data.columns = ['Month-Year', 'Mean_Score', 'Median_Score', 'Raid_Count', 'Total_Tons']
# Filter to months with at least 5 raids
monthly_data = monthly_data[monthly_data['Raid_Count'] >= 5]

# Sort by chronological order
monthly_data['Sort_Order'] = pd.to_datetime(monthly_data['Month-Year'], format='%Y-%m', errors='coerce')
monthly_data = monthly_data.dropna(subset=['Sort_Order'])  # Drop any rows with invalid dates
monthly_data = monthly_data.sort_values('Sort_Order')

# Create scatter plot with size representing tonnage
plt.figure(figsize=(16, 8))
scatter = plt.scatter(range(len(monthly_data)), 
                     monthly_data['Mean_Score'],
                     s=monthly_data['Total_Tons']/100,  # Scale down for better visibility
                     c=monthly_data['Mean_Score'],
                     cmap='viridis',
                     alpha=0.7)

plt.plot(range(len(monthly_data)), monthly_data['Mean_Score'], 'k--', alpha=0.5)
plt.colorbar(scatter, label='Mean Area Bombing Score')

plt.title('Monthly Progression of Area Bombing Scores (USAAF)', fontsize=18)
plt.xlabel('Month-Year', fontsize=14)
plt.ylabel('Mean Area Bombing Score', fontsize=14)
plt.xticks(range(len(monthly_data)), monthly_data['Month-Year'], rotation=90, fontsize=10)
plt.ylim(0, 10)
plt.grid(True, alpha=0.3)

# Add annotation for significant points (high or low scores)
for i, row in enumerate(monthly_data.itertuples()):
    if row.Mean_Score > 8 or row.Mean_Score < 2:
        plt.annotate(f"{row.Mean_Score:.1f}\n({row.Raid_Count} raids)",
                    (i, row.Mean_Score),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.tight_layout()
plt.savefig('plots/usaaf/general/monthly_score_progression.png', dpi=300)
plt.close()

# 5. Tonnage Analysis - Relationship with Area Bombing Score
plt.figure(figsize=(14, 10))

# Create a scatter plot with hexbin for density
hex_plot = plt.hexbin(df['TOTAL_TONS'].clip(0, 500), df['AREA_BOMBING_SCORE_NORMALIZED'], 
                     gridsize=30, cmap='viridis', mincnt=1)
plt.colorbar(hex_plot, label='Count')

# Add trend line
x = df['TOTAL_TONS'].clip(0, 500)
y = df['AREA_BOMBING_SCORE_NORMALIZED']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(np.linspace(0, 500, 100), p(np.linspace(0, 500, 100)), "r--", 
         label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}")

# Average scores by tonnage bins
tonnage_bins = [0, 50, 100, 200, 300, 400, 500]
df['TONNAGE_BIN'] = pd.cut(df['TOTAL_TONS'].clip(0, 500), bins=tonnage_bins)
bin_avgs = df.groupby('TONNAGE_BIN')['AREA_BOMBING_SCORE_NORMALIZED'].mean()

bin_centers = [(tonnage_bins[i] + tonnage_bins[i+1])/2 for i in range(len(tonnage_bins)-1)]
plt.plot(bin_centers, bin_avgs, 'go-', label='Bin Averages')

plt.title('Relationship Between Bombing Tonnage and Area Bombing Score', fontsize=18)
plt.xlabel('Total Tons Dropped (clipped at 500)', fontsize=14)
plt.ylabel('Area Bombing Score', fontsize=14)
plt.ylim(0, 10)
plt.xlim(0, 500)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/usaaf/general/tonnage_vs_score_relationship.png', dpi=300)
plt.close()

# 6. HE vs Incendiary Analysis by Target Category
plt.figure(figsize=(16, 10))

# Calculate total tonnage by category and bomb type
category_bombing = df.groupby('CATEGORY').agg({
    'HE_TONS': 'sum',
    'INCENDIARY_TONS': 'sum',
    'TOTAL_TONS': 'sum',
    'AREA_BOMBING_SCORE_NORMALIZED': 'mean'
}).reset_index()

# Filter to top categories by tonnage
top_categories_by_tonnage = category_bombing.nlargest(10, 'TOTAL_TONS')
# Sort by area bombing score
top_categories_by_tonnage = top_categories_by_tonnage.sort_values('AREA_BOMBING_SCORE_NORMALIZED', ascending=False)

# Create stacked bar chart
bar_width = 0.7
x = np.arange(len(top_categories_by_tonnage))
plt.bar(x, top_categories_by_tonnage['HE_TONS'], 
       color='steelblue', width=bar_width, label='HE Bombs')
plt.bar(x, top_categories_by_tonnage['INCENDIARY_TONS'], 
       bottom=top_categories_by_tonnage['HE_TONS'], color='darkorange', width=bar_width, 
       label='Incendiary Bombs')

# Add area bombing score line (dual y-axis)
ax2 = plt.twinx()
ax2.plot(x, top_categories_by_tonnage['AREA_BOMBING_SCORE_NORMALIZED'], 'ro-', label='Area Bombing Score')
ax2.set_ylim(0, 10)
ax2.set_ylabel('Average Area Bombing Score', fontsize=14, color='darkred')
ax2.tick_params(axis='y', colors='darkred')

# Set x-axis labels
plt.xticks(x, top_categories_by_tonnage['CATEGORY'], rotation=45, ha='right', fontsize=12)
plt.xlabel('Target Category', fontsize=14)
plt.title('HE vs Incendiary Bombing by Target Category (USAAF)', fontsize=18)

# Add two legends
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/usaaf/general/he_vs_incendiary_by_category.png', dpi=300)
plt.close()

# 7. Tonnage distribution by category 
plt.figure(figsize=(14, 8))
# Create box plots of tonnage by target category
top_categories = category_bombing.nlargest(12, 'TOTAL_TONS')['CATEGORY'].tolist()
category_filtered_df = df[df['CATEGORY'].isin(top_categories)].copy()

plt.figure(figsize=(14, 8))
sns.boxplot(x='CATEGORY', y='TOTAL_TONS', data=category_filtered_df,
           order=top_categories, palette='viridis')
plt.title('Distribution of Bombing Tonnage by Target Category (USAAF)', fontsize=18)
plt.xlabel('Target Category', fontsize=14)
plt.ylabel('Total Tons per Raid', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylim(0, 500)  # Clip at 500 tons for better visualization
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/usaaf/general/tonnage_distribution_by_category.png', dpi=300)
plt.close()

# 8. Schweinfurt-specific analysis
print("Creating Schweinfurt-specific analysis...")
schweinfurt_data = df[df['Location'] == 'SCHWEINFURT'].copy()

if len(schweinfurt_data) > 0:
    # Create a dedicated Schweinfurt directory
    os.makedirs('plots/usaaf/cities/schweinfurt', exist_ok=True)
    
    # A. Schweinfurt raids timeline
    plt.figure(figsize=(14, 8))
    
    # Clean and process date fields - handle mixture of types and ranges
    def clean_day(day_value):
        if pd.isna(day_value):
            return 1
        
        # Convert to string first
        day_str = str(day_value)
        
        # If it contains a hyphen (range), take the first value
        if '-' in day_str:
            day_str = day_str.split('-')[0]
            
        # Try to convert to integer, if fails use 1
        try:
            return int(float(day_str))
        except (ValueError, TypeError):
            return 1
    
    schweinfurt_data['Day'] = schweinfurt_data['DAY'].apply(clean_day)
    schweinfurt_data['Month'] = schweinfurt_data['MONTH'].fillna(1).astype(float).astype(int)
    
    # Create date field
    schweinfurt_data['Date'] = pd.to_datetime(
        (1940 + schweinfurt_data['YEAR'].astype(int)).astype(str) + '-' + 
        schweinfurt_data['Month'].astype(str).str.zfill(2) + '-' + 
        schweinfurt_data['Day'].astype(str).str.zfill(2),
        errors='coerce'  # Handle any invalid dates
    )
    
    # Drop any rows with invalid dates
    schweinfurt_data = schweinfurt_data.dropna(subset=['Date'])
    schweinfurt_data = schweinfurt_data.sort_values('Date')
    
    # Create scatter plot of raids
    plt.scatter(range(len(schweinfurt_data)), schweinfurt_data['AREA_BOMBING_SCORE_NORMALIZED'], 
               s=schweinfurt_data['TOTAL_TONS']*2, c=schweinfurt_data['INCENDIARY_PERCENT'], 
               cmap='inferno', alpha=0.8)
    plt.colorbar(label='Incendiary Percentage')
    
    # Add connecting line
    plt.plot(range(len(schweinfurt_data)), schweinfurt_data['AREA_BOMBING_SCORE_NORMALIZED'], 
            'k--', alpha=0.5)
    
    # Add date labels
    date_labels = schweinfurt_data['Date'].dt.strftime('%b %d, %Y')
    plt.xticks(range(len(schweinfurt_data)), date_labels, rotation=45, ha='right', fontsize=10)
    
    # Add annotation with tonnage
    for i, row in enumerate(schweinfurt_data.itertuples()):
        plt.annotate(f"{row.TOTAL_TONS:.1f} tons",
                    (i, row.AREA_BOMBING_SCORE_NORMALIZED),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.title('Schweinfurt Bombing Raids Timeline (USAAF)', fontsize=18)
    plt.xlabel('Raid Date', fontsize=14)
    plt.ylabel('Area Bombing Score', fontsize=14)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/usaaf/cities/schweinfurt/raids_timeline.png', dpi=300)
    plt.close()
    
    # B. Schweinfurt vs Other Ball Bearing Factories comparison
    # Try to find other ball bearing targets
    bearing_targets = df[df['target_name'].str.contains('BEAR|BALL|ROLLER', case=False, na=False)]
    # Filter out Schweinfurt itself
    other_bearing_targets = bearing_targets[bearing_targets['Location'] != 'SCHWEINFURT']
    
    if len(other_bearing_targets) > 0:
        # Comparison data
        comparison_data = pd.DataFrame({
            'Location': ['Schweinfurt', 'Other Bearing Factories'],
            'Avg_Score': [schweinfurt_data['AREA_BOMBING_SCORE_NORMALIZED'].mean(),
                         other_bearing_targets['AREA_BOMBING_SCORE_NORMALIZED'].mean()],
            'Avg_Tonnage': [schweinfurt_data['TOTAL_TONS'].mean(),
                           other_bearing_targets['TOTAL_TONS'].mean()],
            'Avg_Incendiary': [schweinfurt_data['INCENDIARY_PERCENT'].mean(),
                              other_bearing_targets['INCENDIARY_PERCENT'].mean()],
            'Raid_Count': [len(schweinfurt_data), len(other_bearing_targets)]
        })
        
        # Create comparison chart
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(comparison_data))
        bar_width = 0.35
        
        # Plot bombing scores on primary y-axis
        bars1 = ax1.bar(x - bar_width/2, comparison_data['Avg_Score'], 
                       bar_width, label='Avg Area Bombing Score', color='steelblue')
        ax1.set_ylabel('Area Bombing Score', color='steelblue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(0, 10)
        
        # Add a second y-axis for tonnage
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + bar_width/2, comparison_data['Avg_Tonnage'], 
                       bar_width, label='Avg Tonnage per Raid', color='darkorange')
        ax2.set_ylabel('Average Tonnage', color='darkorange', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        # Add raid count
        for i, count in enumerate(comparison_data['Raid_Count']):
            plt.annotate(f"{count} raids",
                       (i, comparison_data['Avg_Tonnage'][i] + 5),
                       ha='center',
                       va='bottom',
                       fontsize=10,
                       color='black')
        
        # Set x-axis labels
        plt.xticks(x, comparison_data['Location'], fontsize=12)
        plt.title('Schweinfurt vs Other Bearing Factory Targets (USAAF)', fontsize=18)
        
        # Add two legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('plots/usaaf/cities/schweinfurt/schweinfurt_vs_other_bearings.png', dpi=300)
        plt.close()

# 9. Advanced Temporal Analysis
print("Creating advanced temporal analysis charts...")

# A. Evolution of bombing characteristics over time (multi-metrics)
plt.figure(figsize=(16, 12))

# Create quarterly data to smooth trends, handling mixed types
df['Clean_Month'] = df['MONTH'].fillna(1).astype(float).astype(int)
df['Quarter_Date'] = pd.to_datetime(
    (1940 + df['YEAR'].astype(int)).astype(str) + '-' + 
    df['Clean_Month'].astype(str).str.zfill(2) + '-01',
    errors='coerce'  # Handle any invalid dates
)
df['Quarter'] = pd.PeriodIndex(df['Quarter_Date'].dropna(), freq='Q')

# Group by quarter
quarterly_data = df.groupby('Quarter').agg({
    'AREA_BOMBING_SCORE_NORMALIZED': 'mean',
    'TOTAL_TONS': 'mean',
    'INCENDIARY_PERCENT': 'mean',
    'raid_id': 'count'
}).reset_index()

quarterly_data.columns = ['Quarter', 'Avg_Score', 'Avg_Tonnage', 'Avg_Incendiary', 'Raid_Count']
quarterly_data = quarterly_data[quarterly_data['Raid_Count'] >= 10]  # Filter to quarters with enough data

# Create a 4-panel plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Area Bombing Score trend
axes[0, 0].plot(range(len(quarterly_data)), quarterly_data['Avg_Score'], 'o-', color='darkblue', linewidth=2)
axes[0, 0].set_title('Area Bombing Score Trend', fontsize=14)
axes[0, 0].set_ylabel('Average Score', fontsize=12)
axes[0, 0].set_ylim(0, 10)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Tonnage trend
axes[0, 1].plot(range(len(quarterly_data)), quarterly_data['Avg_Tonnage'], 'o-', color='darkgreen', linewidth=2)
axes[0, 1].set_title('Average Tonnage Trend', fontsize=14)
axes[0, 1].set_ylabel('Average Tons per Raid', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Incendiary percent trend
axes[1, 0].plot(range(len(quarterly_data)), quarterly_data['Avg_Incendiary'], 'o-', color='darkorange', linewidth=2)
axes[1, 0].set_title('Incendiary Percentage Trend', fontsize=14)
axes[1, 0].set_ylabel('Average Incendiary %', fontsize=12)
axes[1, 0].set_ylim(0, 100)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Raid count trend
axes[1, 1].plot(range(len(quarterly_data)), quarterly_data['Raid_Count'], 'o-', color='darkred', linewidth=2)
axes[1, 1].set_title('Raid Count Trend', fontsize=14)
axes[1, 1].set_ylabel('Number of Raids', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# Set common x-axis labels
for ax in axes.flat:
    ax.set_xticks(range(len(quarterly_data)))
    ax.set_xticklabels([str(q) for q in quarterly_data['Quarter']], rotation=45, ha='right', fontsize=8)

plt.suptitle('Quarterly Evolution of USAAF Bombing Characteristics', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('plots/usaaf/general/quarterly_metrics_evolution.png', dpi=300)
plt.close()

# B. Heat map of area bombing scores and tonnage by year and target category
plt.figure(figsize=(16, 10))

# Create pivot table for area bombing scores
year_category_scores = df.pivot_table(
    values='AREA_BOMBING_SCORE_NORMALIZED',
    index='CATEGORY',
    columns='Year',
    aggfunc='mean'
)

# Filter to include only years 1942-1945 and categories with enough data
year_cols = [y for y in range(1942, 1946) if y in year_category_scores.columns]
year_category_scores = year_category_scores[year_cols]
year_category_scores = year_category_scores.dropna(thresh=len(year_cols)//2)  # Keep rows with at least half the years

# Sort by overall average score (highest to lowest)
year_category_scores['Avg'] = year_category_scores.mean(axis=1)
year_category_scores = year_category_scores.sort_values('Avg', ascending=False).drop('Avg', axis=1)

plt.figure(figsize=(16, 12))
sns.heatmap(year_category_scores, annot=True, fmt='.1f', cmap='viridis', 
           linewidths=0.5, vmin=0, vmax=10, cbar_kws={'label': 'Area Bombing Score'})
plt.title('Evolution of Area Bombing Scores by Target Category and Year (USAAF)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Target Category', fontsize=14)
plt.tight_layout()
plt.savefig('plots/usaaf/general/year_category_score_heatmap.png', dpi=300)
plt.close()

print("USAAF visualization complete. Results saved to plots/usaaf/ directory.") 