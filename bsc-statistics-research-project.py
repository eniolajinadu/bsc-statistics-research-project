# Import all necessary libraries
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add XGBoost and SHAP imports
from xgboost import XGBRegressor
import shap
import contextily as ctx

# 1. Data Loading Function
def load_all_data(data_folder_path):
    """Load all CSV and GeoJSON files from a folder"""
    data_folder = Path(data_folder_path)
    all_dataframes = {}
    
    # Load all CSV files
    csv_files = list(data_folder.glob("*.csv"))
    for file_path in csv_files:
        df_name = file_path.stem
        try:
            all_dataframes[df_name] = pd.read_csv(file_path, on_bad_lines='warn', encoding='latin-1')
            print(f"Successfully loaded CSV: {df_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Load all GeoJSON files
    geojson_files = list(data_folder.glob("*.geojson"))
    for file_path in geojson_files:
        df_name = file_path.stem
        try:
            all_dataframes[df_name] = gpd.read_file(file_path)
            print(f"Successfully loaded GeoJSON: {df_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_dataframes

# 2. Data Exploration and Integration
def explore_and_integrate_data(all_dataframes):
    """Explore and integrate all available data sources"""
    print("=== DATA EXPLORATION AND INTEGRATION ===")
    
    # Create a comprehensive data dictionary
    data_sources = {
        'Population': ['lagos_population', '2006_population', 'worldpop'],
        'Land Cover': ['1984-2013_spatial_change', '1984-2020_lulc_area', '2013-2024_spatial_change', 
                      'built-up-areas', 'farmlands'],
        'Transportation': ['brt_buses_passengers', 'lagbus_buses_passengers', 'lagbus_fleet_passengers',
                          'lagbus_operations', 'lagbus_route_info', 'lagbus_targets_actuals',
                          'lagbus_vehicle_registrations', 'ferry_jetties_passengers', 
                          'ferry_routes_passengers', 'rail_cleaned', 'road_cleaned',
                          'vehicles_by_country', 'vehicles_by_make', 'vehicles_by_ownership', 
                          'vehicles_by_year'],
        'Infrastructure': ['ambulance-ememergency-services', 'churches', 'dump-sites',
                          'electricity-sub-stations', 'environmental-sites', 'factoriesindustrial-sites',
                          'filling-stations', 'fire-station', 'government-buildings',
                          'health-care-facilities-primary-secondary-and-tertiary', 'health_facilities',
                          'hospitals_2019', 'hospitals_2_2019', 'idp-sites', 'laboratories',
                          'markets', 'mosques', 'pharmaceutical-facilities', 'police-stations',
                          'post-offices', 'primary-schools', 'private-schools', 'public-schools',
                          'public-water-points', 'public_jss', 'public_primary_school',
                          'public_sss', 'religious-schools', 'secondary-schools', 'settlement-points',
                          'small-settlement-areas', 'small-settlement-points', 'tertiary-schools',
                          'water_facilities'],
        'Environmental': ['climate', 'emissions', 'energy', 'freshwater', 'greenhouse_gas_emissions',
                         'sustainability'],
        'Socioeconomic': ['cpi_index', 'food_cpi_index', 'cpi_coicop_classification', 
                         'Health_Risk_factors', 'health_system'],
        'Landscape Metrics': ['2000-2010_class_landscape_metrics', '2000-2010_shannon_index',
                             'landscape_metrics'],
        'Boundaries': ['local-government-administrative-boundaries', 'state-administrative-boundaries',
                      'operational-ward-boundaries']
    }
    
    # Check availability of each data source
    available_data = {}
    for category, sources in data_sources.items():
        available_sources = [s for s in sources if s in all_dataframes]
        if available_sources:
            available_data[category] = available_sources
            print(f"\n{category}:")
            for source in available_sources:
                df = all_dataframes[source]
                print(f"  - {source}: {df.shape}, Columns: {list(df.columns)[:3]}...")
    
    return available_data

# 3. Base DataFrame Creation with Enhanced Features
def create_enhanced_base_dataframe(all_dataframes, available_data):
    """Create a base dataframe with enhanced features"""
    print("\n=== CREATING ENHANCED BASE DATAFRAME ===")
    
    # Get LGA names from boundaries
    if 'local-government-administrative-boundaries' in all_dataframes:
        lga_df = all_dataframes['local-government-administrative-boundaries']
        lga_columns = [col for col in lga_df.columns if any(x in col.lower() for x in ['name', 'lga', 'admin'])]
        
        if lga_columns:
            lga_names = lga_df[lga_columns[0]].unique()
            print(f"Found {len(lga_names)} LGAs")
            
            # Calculate LGA areas
            lga_df = lga_df.copy()
            lga_df['area_sqkm'] = lga_df.geometry.area / 10**6  # Convert to sq km
            lga_areas = lga_df.set_index(lga_columns[0])['area_sqkm'].to_dict()
        else:
            lga_names = [f"LGA_{i}" for i in range(1, 21)]
            lga_areas = {lga: np.random.uniform(50, 200) for lga in lga_names}
            print("Using default LGA names and areas")
    else:
        lga_names = [f"LGA_{i}" for i in range(1, 21)]
        lga_areas = {lga: np.random.uniform(50, 200) for lga in lga_names}
        print("Using default LGA names and areas")
    
    # Get available years from all datasets
    all_years = set()
    for category, sources in available_data.items():
        for source in sources:
            df = all_dataframes[source]
            if 'year' in df.columns:
                all_years.update(df['year'].dropna().unique())
            elif 'Year' in df.columns:
                all_years.update(df['Year'].dropna().unique())
    
    # Use 5-year intervals if we have many years
    if len(all_years) > 10:
        years = sorted([y for y in all_years if 1980 <= y <= 2024])
        selected_years = years[::5]  # Every 5 years
        if 2024 not in selected_years:
            selected_years.append(2024)
    else:
        selected_years = sorted(all_years)
    
    print(f"Selected years: {selected_years}")
    
    # Create base dataframe
    base_df = pd.DataFrame([(lga, year) for lga in lga_names for year in selected_years], 
                          columns=['LGA', 'year'])
    
    # Add LGA area
    base_df['area_sqkm'] = base_df['LGA'].map(lga_areas)
    
    print(f"Created base dataframe with {len(base_df)} rows")
    return base_df

# 4. Land Cover Processing Function
def process_land_cover_data(base_df, all_dataframes):
    """Process land cover data and add to base dataframe"""
    print("\n=== PROCESSING LAND COVER DATA ===")
    
    # First, let's examine what land cover data we have
    land_cover_sources = [
        '1984-2013_spatial_change', '1984-2020_lulc_area', '2013-2024_spatial_change',
        'built-up-areas', 'farmlands'
    ]
    
    # Check which datasets are available and their structure
    available_land_cover = {}
    for source in land_cover_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            print(f"Examining {source}: shape {df.shape}, columns {list(df.columns)}")
            available_land_cover[source] = df
    
    # Try to extract built-up area data from different dataset structures
    builtup_data = []
    
    # Approach 1: Try to use the lulc_area dataset which has Year and Area by Land Use Type
    if '1984-2020_lulc_area' in available_land_cover:
        lulc_df = available_land_cover['1984-2020_lulc_area']
        print("Processing 1984-2020_lulc_area dataset")
        
        # Check if this dataset has the structure we need
        if all(col in lulc_df.columns for col in ['Year', 'Land Use Type', 'Area (ha)']):
            # Filter for built-up areas
            builtup_types = ['Built-up', 'Built up', 'Urban', 'Built-up Area', 'Artificial surfaces']
            builtup_df = lulc_df[lulc_df['Land Use Type'].isin(builtup_types)]
            
            if not builtup_df.empty:
                # Pivot to get area by year
                builtup_by_year = builtup_df.groupby('Year')['Area (ha)'].sum().reset_index()
                builtup_by_year.columns = ['year', 'BuiltUp_area']
                
                # For this dataset, we don't have LGA-level data, so we'll assign to all LGAs
                for lga in base_df['LGA'].unique():
                    for _, row in builtup_by_year.iterrows():
                        builtup_data.append({
                            'LGA': lga,
                            'year': row['year'],
                            'BuiltUp_area': row['BuiltUp_area']
                        })
                print("Extracted built-up area data from 1984-2020_lulc_area")
    
    # Approach 2: Try to use spatial change datasets
    for change_source in ['1984-2013_spatial_change', '2013-2024_spatial_change']:
        if change_source in available_land_cover:
            change_df = available_land_cover[change_source]
            print(f"Processing {change_source} dataset")
            
            # These datasets seem to have year columns with area values
            year_columns = [col for col in change_df.columns if 'Area' in col and '(' in col and ')' in col]
            for col in year_columns:
                # Extract year from column name
                year_str = col.split(' ')[0]
                try:
                    year = int(year_str)
                    # Look for built-up land use class
                    builtup_rows = change_df[change_df['LULC Class'].str.contains('Built-up|Built up|Urban', case=False, na=False)]
                    if not builtup_rows.empty:
                        builtup_area = builtup_rows[col].values[0]
                        
                        # Assign to all LGAs (since dataset doesn't have LGA breakdown)
                        for lga in base_df['LGA'].unique():
                            builtup_data.append({
                                'LGA': lga,
                                'year': year,
                                'BuiltUp_area': builtup_area
                            })
                        print(f"Extracted built-up area for {year} from {change_source}")
                except ValueError:
                    continue
    
    # If no proper land cover data found, create placeholder data
    if not builtup_data:
        print("No proper land cover data found. Creating placeholder data.")
        for lga in base_df['LGA'].unique():
            for year in base_df['year'].unique():
                # Create synthetic data that increases over time
                base_year = 1984
                growth_rate = 0.05  # 5% annual growth
                builtup_area = 100 * (1 + growth_rate) ** (year - base_year)
                builtup_data.append({
                    'LGA': lga,
                    'year': year,
                    'BuiltUp_area': builtup_area
                })
    
    # Create DataFrame from builtup_data
    land_cover_df = pd.DataFrame(builtup_data)
    
    # Merge with base dataframe
    base_df = pd.merge(base_df, land_cover_df, on=['LGA', 'year'], how='left')
    
    return base_df

# 5. Enhanced Data Integration with Spatial Joins
def integrate_enhanced_data(base_df, all_dataframes, available_data):
    """Integrate enhanced data with spatial joins and feature engineering"""
    print("\n=== INTEGRATING ENHANCED DATA ===")
    
    # Get LGA boundaries for spatial joins
    if 'local-government-administrative-boundaries' in all_dataframes:
        lga_gdf = all_dataframes['local-government-administrative-boundaries']
        lga_columns = [col for col in lga_gdf.columns if any(x in col.lower() for x in ['name', 'lga', 'admin'])]
        
        if lga_columns:
            lga_gdf = lga_gdf.set_index(lga_columns[0])
    
    # Add infrastructure counts via spatial joins
    infrastructure_sources = available_data.get('Infrastructure', [])
    for source in infrastructure_sources:
        if source in all_dataframes and 'geometry' in all_dataframes[source].columns:
            infra_gdf = all_dataframes[source]
            
            # Count points per LGA
            if hasattr(lga_gdf, 'geometry'):
                try:
                    # Spatial join to count points in each LGA
                    joined = gpd.sjoin(infra_gdf, lga_gdf, how='inner', predicate='within')
                    counts = joined.groupby(lga_columns[0]).size().reset_index(name=f'{source}_count')
                    counts.rename(columns={lga_columns[0]: 'LGA'}, inplace=True)
                    
                    # Add to base dataframe (assuming counts are time-invariant for now)
                    base_df = pd.merge(base_df, counts, on='LGA', how='left')
                    print(f"Added {source} count data")
                except Exception as e:
                    print(f"Could not spatially join {source}: {e}")
    
    # 3. Add other data sources
    # Population
    pop_sources = available_data.get('Population', [])
    for source in pop_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            if 'LGA' in df.columns and 'year' in df.columns and 'population' in df.columns:
                base_df = pd.merge(base_df, df[['LGA', 'year', 'population']], on=['LGA', 'year'], how='left')
                print(f"Added population data from {source}")
                break
            elif 'lga' in df.columns and 'year' in df.columns and 'population' in df.columns:
                base_df = pd.merge(base_df, df[['lga', 'year', 'population']].rename(columns={'lga': 'LGA'}), 
                                  on=['LGA', 'year'], how='left')
                print(f"Added population data from {source}")
                break
    
    # Transportation
    transport_sources = available_data.get('Transportation', [])
    for source in transport_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            # Find numeric columns that aren't identifiers
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                           and col not in ['LGA', 'lga', 'year', 'id']]
            
            if numeric_cols and ('LGA' in df.columns or 'lga' in df.columns) and 'year' in df.columns:
                id_col = 'LGA' if 'LGA' in df.columns else 'lga'
                cols_to_merge = [id_col, 'year'] + numeric_cols[:2]  # Take first 2 numeric columns
                base_df = pd.merge(base_df, df[cols_to_merge].rename(columns={id_col: 'LGA'}), 
                                  on=['LGA', 'year'], how='left')
                print(f"Added transportation data from {source}")
    
    # Environmental
    env_sources = available_data.get('Environmental', [])
    for source in env_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            if 'LGA' in df.columns and 'year' in df.columns:
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                               and col not in ['LGA', 'year', 'id']]
                if numeric_cols:
                    cols_to_merge = ['LGA', 'year'] + numeric_cols[:2]
                    base_df = pd.merge(base_df, df[cols_to_merge], on=['LGA', 'year'], how='left')
                    print(f"Added environmental data from {source}")
    
    # Socioeconomic
    socio_sources = available_data.get('Socioeconomic', [])
    for source in socio_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            if 'LGA' in df.columns and 'year' in df.columns:
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                               and col not in ['LGA', 'year', 'id']]
                if numeric_cols:
                    cols_to_merge = ['LGA', 'year'] + numeric_cols[:2]
                    base_df = pd.merge(base_df, df[cols_to_merge], on=['LGA', 'year'], how='left')
                    print(f"Added socioeconomic data from {source}")
    
    # Landscape Metrics
    landscape_sources = available_data.get('Landscape Metrics', [])
    for source in landscape_sources:
        if source in all_dataframes:
            df = all_dataframes[source]
            if 'LGA' in df.columns and 'year' in df.columns:
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                               and col not in ['LGA', 'year', 'id']]
                if numeric_cols:
                    cols_to_merge = ['LGA', 'year'] + numeric_cols[:2]
                    base_df = pd.merge(base_df, df[cols_to_merge], on=['LGA', 'year'], how='left')
                    print(f"Added landscape metrics from {source}")
    
    return base_df

# 6. Advanced Feature Engineering
def engineer_advanced_features(base_df):
    """Create advanced features including lagged variables, interactions, and ratios"""
    print("\n=== ENGINEERING ADVANCED FEATURES ===")
    
    # Sort by LGA and year
    base_df.sort_values(['LGA', 'year'], inplace=True)
    
    # 1. Create lagged features for key variables
    lag_periods = [1]  # Use only 1 period lag for now
    
    for lag in lag_periods:
        if 'BuiltUp_area' in base_df.columns:
            base_df[f'BuiltUp_area_lag_{lag}'] = base_df.groupby('LGA')['BuiltUp_area'].shift(lag)
    
    # 2. Calculate growth rates if we have the required data
    if 'BuiltUp_area' in base_df.columns and 'BuiltUp_area_lag_1' in base_df.columns:
        # Handle division by zero
        base_df['BuiltUp_growth_pct'] = (
            (base_df['BuiltUp_area'] - base_df['BuiltUp_area_lag_1']) / 
            base_df['BuiltUp_area_lag_1'].replace(0, np.nan) * 100
        )
    else:
        print("Warning: Could not create BuiltUp_growth_pct - missing required columns")
    
    # 3. Create density measures
    if 'area_sqkm' in base_df.columns and 'BuiltUp_area' in base_df.columns:
        base_df['builtup_density'] = base_df['BuiltUp_area'] / base_df['area_sqkm']
    
    # 4. Add population data if available
    if 'population' in base_df.columns:
        if 'area_sqkm' in base_df.columns:
            base_df['population_density'] = base_df['population'] / base_df['area_sqkm']
        
        # Create lagged population
        for lag in lag_periods:
            base_df[f'population_lag_{lag}'] = base_df.groupby('LGA')['population'].shift(lag)
        
        # Calculate population growth
        if 'population_lag_1' in base_df.columns:
            base_df['population_growth_pct'] = (
                (base_df['population'] - base_df['population_lag_1']) / 
                base_df['population_lag_1'].replace(0, np.nan) * 100
            )
    
    return base_df

# 7. Handle Missing Data
def handle_missing_data(base_df):
    """Handle missing values in the dataset while protecting target variables"""
    print("\n=== HANDLING MISSING DATA ===")
    
    # Check missing data percentage
    missing_pct = base_df.isnull().mean() * 100
    print("Missing data percentage by column:")
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")
    
    # Protect target variable and its dependencies from removal
    protected_cols = ['BuiltUp_area', 'BuiltUp_area_lag_1', 'BuiltUp_growth_pct', 'builtup_density']
    
    # Remove columns with too much missing data (except protected columns)
    high_missing_cols = [col for col in missing_pct[missing_pct > 50].index 
                        if col not in protected_cols]
    base_df.drop(columns=high_missing_cols, inplace=True)
    print(f"Removed columns with >50% missing data: {high_missing_cols}")
    
    # For time-series data, use forward/backward fill within each LGA
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        base_df[col] = base_df.groupby('LGA')[col].transform(
            lambda x: x.ffill().bfill()
        )
    
    # For remaining missing values, use median imputation
    for col in numeric_cols:
        if base_df[col].isnull().any():
            base_df[col].fillna(base_df[col].median(), inplace=True)
    
    print("Missing data handling complete")
    return base_df

# 8. Prepare Data for ML
def prepare_ml_data(base_df, target_var='BuiltUp_growth_pct'):
    """Prepare data for machine learning"""
    print("\n=== PREPARING DATA FOR MACHINE LEARNING ===")
    
    # Check if target variable exists
    if target_var not in base_df.columns:
        print(f"ERROR: Target variable '{target_var}' not found in dataframe.")
        print("Available columns:", base_df.columns.tolist())
        return base_df, None, None, None
    
    # Remove rows where target is missing
    valid_indices = base_df[target_var].notna()
    base_df = base_df[valid_indices]
    
    if len(base_df) == 0:
        print(f"ERROR: No rows with valid target variable '{target_var}'.")
        return base_df, None, None, None
    
    # Select features (all numeric columns except identifiers and target)
    exclude_cols = ['LGA', 'year']
    if target_var in base_df.columns:
        exclude_cols.append(target_var)
    
    feature_cols = [col for col in base_df.columns 
                   if col not in exclude_cols and base_df[col].dtype in ['float64', 'int64']]
    
    X = base_df[feature_cols]
    y = base_df[target_var]
    
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    
    return base_df, X, y, feature_cols

# 9. Multiple Modeling Approaches - MODIFIED VERSION
def train_multiple_models(final_df, X, y, feature_cols):
    """Train and evaluate multiple modeling approaches"""
    print("\n=== TRAINING MULTIPLE MODELS ===")
    
    # Temporal split: train <= 2018, test > 2018
    train_idx = final_df["year"] <= 2018
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[~train_idx], y[~train_idx]
    
    print(f"Training set: {X_train.shape} (years <= 2018)")
    print(f"Test set: {X_test.shape} (years > 2018)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to try
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Ridge Regression': Ridge(alpha=0.1, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (R²: {results[best_model_name]['R2']:.4f})")
    
    # Cross-validation for best model
    print(f"\nCross-validation for {best_model_name}:")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        
        # Scale features
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler.transform(X_test_cv)
        
        best_model.fit(X_train_cv_scaled, y_train_cv)
        pred_cv = best_model.predict(X_test_cv_scaled)
        cv_scores.append(r2_score(y_test_cv, pred_cv))
    
    print(f"CV R² scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    # Return the scaled test data for SHAP analysis
    return best_model, scaler, results, best_model_name, X_test_scaled, y_test, train_idx

# 10. SHAP Analysis with Enhanced Visualizations - MODIFIED VERSION
def perform_shap_analysis(model, X_test_scaled, feature_cols, final_df, y_test, train_idx):
    """Perform SHAP analysis with enhanced visualizations"""
    print("\n=== SHAP ANALYSIS ===")
    
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test_scaled)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, show=False)
        plt.title('SHAP Feature Importance - Drivers of Urban Growth')
        plt.tight_layout()
        plt.savefig('./results/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title('SHAP Beeswarm Plot - Impact of Features on Urban Growth')
        plt.tight_layout()
        plt.savefig('./results/shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot
        plt.figure(fagure=(12, 8))
        shap.plots.bar(shap_values, show=False)
        plt.title('SHAP Bar Plot - Mean Absolute Impact of Features')
        plt.tight_layout()
        plt.savefig('./results/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a dataframe with SHAP values for further analysis
        shap_df = pd.DataFrame(shap_values.values, columns=[f'shap_{col}' for col in feature_cols])
        shap_df['LGA'] = final_df[~train_idx]['LGA'].values
        shap_df['year'] = final_df[~train_idx]['year'].values
        shap_df['actual'] = y_test.values
        shap_df['predicted'] = model.predict(X_test_scaled)
        
        # Save SHAP values
        shap_df.to_csv('./results/shap_values.csv', index=False)
        
        return shap_df
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("This might be due to large dataset size or memory constraints")
        return None

# 11. Spatial Analysis and Visualization
def perform_spatial_analysis(final_df, all_dataframes, model, X, y, feature_cols, scaler):
    """Perform spatial analysis and create maps"""
    print("\n=== SPATIAL ANALYSIS ===")
    
    # Get LGA boundaries
    if 'local-government-administrative-boundaries' in all_dataframes:
        lga_gdf = all_dataframes['local-government-administrative-boundaries']
        lga_columns = [col for col in lga_gdf.columns if any(x in col.lower() for x in ['name', 'lga', 'admin'])]
        
        if lga_columns:
            lga_gdf = lga_gdf.set_index(lga_columns[0])
            
            # Calculate predicted values for all LGAs and years
            final_df['predicted_growth'] = model.predict(scaler.transform(X))
            
            # Create a map of predicted growth for the most recent year
            recent_year = final_df['year'].max()
            recent_data = final_df[final_df['year'] == recent_year]
            
            # Merge with LGA boundaries
            map_data = lga_gdf.merge(recent_data[['LGA', 'predicted_growth']], 
                                    left_index=True, right_on='LGA', how='left')
            
            # Create map
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            map_data.plot(column='predicted_growth', cmap='RdYlGn', legend=True, 
                         ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Predicted Urban Growth by LGA ({recent_year})', fontsize=16)
            ax.set_axis_off()
            
            # Add basemap
            try:
                ctx.add_basemap(ax, crs=map_data.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            except:
                pass
            
            plt.tight_layout()
            plt.savefig('./results/predicted_growth_map.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Calculate Moran's I for spatial autocorrelation (if we have spatial weights)
    try:
        from esda.moran import Moran
        from libpysal.weights import Queen
        
        # Create spatial weights
        w = Queen.from_dataframe(lga_gdf)
        w.transform = 'r'
        
        # Calculate Moran's I for actual and predicted growth
        moran_actual = Moran(y, w)
        moran_predicted = Moran(final_df['predicted_growth'], w)
        
        print(f"Moran's I (Actual): {moran_actual.I:.3f} (p-value: {moran_actual.p_sim:.3f})")
        print(f"Moran's I (Predicted): {moran_predicted.I:.3f} (p-value: {moran_predicted.p_sim:.3f})")
        
    except ImportError:
        print("Spatial analysis libraries not available. Install esda and libpysal for spatial autocorrelation analysis.")
    except Exception as e:
        print(f"Spatial autocorrelation analysis failed: {e}")

# 12. Scenario Testing
def perform_scenario_testing(model, scaler, X, final_df, feature_cols):
    """Test different scenarios for urban growth"""
    print("\n=== SCENARIO TESTING ===")
    
    # Create baseline scenario (current values)
    baseline_idx = final_df['year'] == final_df['year'].max()
    X_baseline = X[baseline_idx]
    baseline_pred = model.predict(scaler.transform(X_baseline))
    
    # Scenario 1: 10% population increase
    X_scenario1 = X_baseline.copy()
    if 'population' in feature_cols:
        pop_idx = feature_cols.index('population')
        X_scenario1[:, pop_idx] = X_scenario1[:, pop_idx] * 1.1
    
    scenario1_pred = model.predict(scaler.transform(X_scenario1))
    
    # Scenario 2: 20% increase in transportation infrastructure
    X_scenario2 = X_baseline.copy()
    transport_cols = [i for i, col in enumerate(feature_cols) if any(x in col for x in ['bus', 'vehicle', 'transport'])]
    for idx in transport_cols:
        X_scenario2[:, idx] = X_scenario2[:, idx] * 1.2
    
    scenario2_pred = model.predict(scaler.transform(X_scenario2))
    
    # Compare scenarios
    scenario_results = pd.DataFrame({
        'LGA': final_df[baseline_idx]['LGA'].values,
        'Baseline': baseline_pred,
        'Population_Increase': scenario1_pred,
        'Transport_Improvement': scenario2_pred
    })
    
    # Calculate differences
    scenario_results['Population_Increase_Diff'] = scenario_results['Population_Increase'] - scenario_results['Baseline']
    scenario_results['Transport_Improvement_Diff'] = scenario_results['Transport_Improvement'] - scenario_results['Baseline']
    
    print("Scenario testing results:")
    print(f"Average baseline growth: {scenario_results['Baseline'].mean():.2f}%")
    print(f"Average growth with population increase: {scenario_results['Population_Increase'].mean():.2f}%")
    print(f"Average growth with transport improvement: {scenario_results['Transport_Improvement'].mean():.2f}%")
    
    # Save scenario results
    scenario_results.to_csv('./results/scenario_analysis.csv', index=False)
    
    return scenario_results

# 13. Main Execution - MODIFIED VERSION
def main():
    """Main function to run the complete analysis"""
    print("=== COMPREHENSIVE LAGOS URBANIZATION ANALYSIS ===\n")
    
    # Load all data
    data_folder = "./raw"
    all_dataframes = load_all_data(data_folder)
    
    # Explore and integrate data
    available_data = explore_and_integrate_data(all_dataframes)
    
    # Create base dataframe
    base_df = create_enhanced_base_dataframe(all_dataframes, available_data)
    
    # Process land cover data
    base_df = process_land_cover_data(base_df, all_dataframes)
    
    # Add other data sources
    base_df = integrate_enhanced_data(base_df, all_dataframes, available_data)
    
    # Engineer advanced features
    base_df = engineer_advanced_features(base_df)
    
    # Handle missing data
    base_df = handle_missing_data(base_df)
    
    # Prepare for ML
    final_df, X, y, feature_cols = prepare_ml_data(base_df)
    
    # Check if we have data for modeling
    if y is None or len(y) == 0:
        print("ERROR: No target variable available for modeling.")
        print("Final dataframe shape:", final_df.shape if final_df is not None else "None")
        if final_df is not None:
            print("Final dataframe columns:", final_df.columns.tolist())
        return
    
    # Train and evaluate models
    model, scaler, results, best_model_name, X_test_scaled, y_test, train_idx = train_multiple_models(final_df, X, y, feature_cols)
    
    # Perform SHAP analysis
    shap_df = perform_shap_analysis(model, X_test_scaled, feature_cols, final_df, y_test, train_idx)
    
    # Perform spatial analysis
    perform_spatial_analysis(final_df, all_dataframes, model, X, y, feature_cols, scaler)
    
    # Perform scenario testing
    scenario_results = perform_scenario_testing(model, scaler, X.values, final_df, feature_cols)
    
    print("\nAnalysis complete! Results saved to disk.")

# Run the analysis
if __name__ == "__main__":
    main()