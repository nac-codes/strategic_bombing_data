# USAAF Bombing Campaign Analysis Dashboard

This repository contains a Streamlit dashboard for analyzing United States Army Air Forces (USAAF) bombing campaigns during World War II, with a focus on comparing precision vs. area bombing strategies.

## Deployment Package

This deployment package includes all necessary files to run the USAAF Bombing Campaign Analysis Dashboard:

### Application Files
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `visualize_usaaf_bombing.py` - Reference script for how visualizations were generated

### Data Files
- `combined_attack_data.csv` - Original combined attack data
- `processed_data/usaaf/usaaf_raids_full.csv` - Processed USAAF raids data

### Visualization Files
- `plots/usaaf/` - Directory containing all pre-generated visualizations:
  - `general/` - Overall bombing campaign visualizations
  - `categories/` - Target category-specific visualizations
  - `cities/` - City-specific visualizations
  - `years/` - Year-specific visualizations

## Installation

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Dashboard

Run the Streamlit app:
```
streamlit run app.py
```

The dashboard will be available at http://localhost:8501 in your web browser.

## Dashboard Features

- **General Analysis** - Overall distribution of area bombing scores, tonnage analysis, HE vs. incendiary bombing trends
- **City Analysis** - Filter data by specific cities with visualizations for each target
- **Category Analysis** - Examine bombing patterns by target category
- **Year Analysis** - Study the evolution of bombing strategies by year (1940-1945)
- **Data Download** - Access and download the raw data

## Data Quality Note

This data was derived from historical bombing records and has some limitations:
- Contains OCR errors from original documents
- May include misreadings of original records
- Could have processing/interpretation errors

Despite these limitations, the dataset provides a robust overall picture of USAAF bombing campaigns.

## License

[Specify license information here]

## Acknowledgements

[Add acknowledgements as appropriate] 