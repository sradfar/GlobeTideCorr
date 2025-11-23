# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu)
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified: Nov 22, 2025
#
# Description:
# This script performs an enhanced global analysis of tidal-constituent
# co-variability across long-record tide gauges. It extracts primary and
# secondary tidal constituents (M2, N2, S2, O1, K1, MK3, MS4), computes
# annual maxima, and evaluates station-level relationships using both
# Kendall τ and Pearson r correlation metrics.
#
# Key Outputs:
# - Global Robinson-projection maps of correlation strength and significance.
# - Station-level summary reports including significant co-varying constituents.
# - Statistical comparison between Kendall and Pearson methods, including
#   agreement rates and difference distributions.
# - Consolidated tables describing correlation behavior across the full network.
#
# For methodological details and scientific context, see:
# Radfar, S., Taheri, P., and Moftakhari, H. (2025). Global evidence for
# coherent variability in major tidal constituents. *Environmental Research
# Letters*.
#
# Disclaimer:
# This script is provided for research and educational purposes only and
# supplied “as is” without warranty of any kind. The author assumes no
# responsibility for errors, omissions, or outcomes resulting from its use.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import contextily as ctx
import geopandas as gpd
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Define the correct path to your data folder
DATA_FOLDER = r"...\Station data"

# Define minimum years parameter
MIN_YEARS = 20

# Create output directories
OUTPUT_BASE = "correlation_analysis_results"
KENDALL_DIR = os.path.join(OUTPUT_BASE, "kendall_tau")
PEARSON_DIR = os.path.join(OUTPUT_BASE, "pearson")
COMPARISON_DIR = os.path.join(OUTPUT_BASE, "comparison_analysis")

for directory in [OUTPUT_BASE, KENDALL_DIR, PEARSON_DIR, COMPARISON_DIR]:
    os.makedirs(directory, exist_ok=True)
    # Create subdirectories for maps and reports
    if directory != OUTPUT_BASE:
        os.makedirs(os.path.join(directory, "maps"), exist_ok=True)
        os.makedirs(os.path.join(directory, "reports"), exist_ok=True)

def calculate_correlations(temp_df, primary, secondary, columns, method='kendall'):
    """
    Calculate correlations using specified method (kendall or pearson)
    """
    correlations = {}
    p_values = {}
    
    for col in columns:
        if col == primary:
            # Target constituent IS primary - correlate with secondary
            if secondary and secondary in temp_df.columns:
                if method == 'kendall':
                    corr, p_val = stats.kendalltau(temp_df[primary], temp_df[secondary])
                else:  # pearson
                    corr, p_val = stats.pearsonr(temp_df[primary], temp_df[secondary])
                correlations[col] = corr
                p_values[col] = p_val
        else:
            # Target constituent is NOT primary - correlate primary with target
            if col in temp_df.columns:
                if method == 'kendall':
                    corr, p_val = stats.kendalltau(temp_df[primary], temp_df[col])
                else:  # pearson
                    corr, p_val = stats.pearsonr(temp_df[primary], temp_df[col])
                correlations[col] = corr
                p_values[col] = p_val
    
    return correlations, p_values

def plot_gauges(df_p, df_dep, col, name, subplot_label, output_dir, method_name, latmin=None, latmax=None):
    """
    Plot tidal gauge data using Robinson projection
    """
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': ccrs.Robinson()})

    # Add background features (no ocean color)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    
    # Set global extent for Robinson projection
    ax.set_global()
    
    # Plot data points
    sig = df_p[col] < 0.1
    nonsig = ~sig
    
    # Plot non-significant points (gray circles)
    if nonsig.any():
        ax.scatter(
            df_dep.loc[nonsig, 'lon'],
            df_dep.loc[nonsig, 'lat'],
            s=30,
            color='gray',
            alpha=0.8,
            label='p ≥ 0.1',
            transform=ccrs.PlateCarree(),
            marker='o',
            edgecolor='black',
            linewidth=0.3
        )
    
    # Plot significant points (colored circles)
    if sig.any():
        scatter = ax.scatter(
            df_dep.loc[sig, 'lon'],
            df_dep.loc[sig, 'lat'],
            c=df_dep.loc[sig, col],
            s=80,
            cmap='seismic',
            marker='o',
            edgecolor='black',
            linewidth=0.5,
            label='p < 0.1',
            vmin=-1,
            vmax=1,
            transform=ccrs.PlateCarree()
        )
        
        # Add colorbar
        cb = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
        cb.set_label(f'{method_name} correlation', fontsize=14, fontweight='bold')
    
    # Add subplot label
    ax.text(0.02, 0.98, f'{subplot_label})', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Save figure
    output_path = os.path.join(output_dir, "maps", f"enhanced_dep{col}_{name}_{method_name.lower()}_robinson.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def generate_summary_report(df_dep, df_p, columns, df_max_list, method_name, output_dir):
    """
    Generate comprehensive summary report for correlation analysis
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"{method_name.upper()} CORRELATION ANALYSIS SUMMARY")
    report_lines.append("="*80)
    
    for target_constituent in columns:
        # Count stations for this constituent
        valid_corr = df_dep[target_constituent].notna()
        total_stations = valid_corr.sum()
        
        if total_stations > 0:
            # Get data for this constituent
            correlations_subset = df_dep[valid_corr]
            p_values_subset = df_p[valid_corr]
            
            # Count different correlation types
            primary_is_target = 0
            primary_not_target = 0
            most_common_secondary = None
            secondary_counts = {}
            
            for idx, row in correlations_subset.iterrows():
                gauge_name = row['GaugeName']
                if gauge_name in df_max_list:
                    temp_df = df_max_list[gauge_name]
                    if not temp_df.empty:
                        primary = temp_df['Prime'].iat[0]
                        secondary = temp_df['Secondary'].iat[0] if 'Secondary' in temp_df.columns and not pd.isna(temp_df['Secondary'].iat[0]) else None
                        
                        if primary == target_constituent:
                            primary_is_target += 1
                            if secondary:
                                secondary_counts[secondary] = secondary_counts.get(secondary, 0) + 1
                        else:
                            primary_not_target += 1
            
            # Find most common secondary
            if secondary_counts:
                most_common_secondary = max(secondary_counts, key=secondary_counts.get)
                most_common_count = secondary_counts[most_common_secondary]
            
            # Count significant correlations
            significant = (p_values_subset[target_constituent] < 0.1).sum()
            
            # Calculate correlation statistics
            corr_values = correlations_subset[target_constituent].dropna()
            
            report_lines.append(f"\n{target_constituent} CORRELATIONS ({total_stations} total stations):")
            report_lines.append(f"  - When {target_constituent} is PRIMARY: {primary_is_target} stations ({target_constituent} vs secondary)")
            report_lines.append(f"  - When {target_constituent} is NOT primary: {primary_not_target} stations (primary vs {target_constituent})")
            report_lines.append(f"  - Significant correlations (p < 0.1): {significant} stations ({significant/total_stations*100:.1f}%)")
            report_lines.append(f"  - Mean correlation: {corr_values.mean():.3f}")
            report_lines.append(f"  - Std correlation: {corr_values.std():.3f}")
            report_lines.append(f"  - Min/Max correlation: {corr_values.min():.3f} / {corr_values.max():.3f}")
            
            if most_common_secondary:
                report_lines.append(f"  - Most common secondary when {target_constituent} is primary: {most_common_secondary} ({most_common_count} stations)")
    
    # Save report
    report_path = os.path.join(output_dir, "reports", f"{method_name.lower()}_correlation_summary.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_lines

def create_significance_report(df_dep, df_p, columns, df_max_list, method_name, output_dir):
    """
    Create detailed significance report for each gauge
    """
    significance_data = []

    for idx, row in df_dep.iterrows():
        gauge_name = row['GaugeName']
        lat = row['lat']
        lon = row['lon']
        years = row['years']
        
        significant_constituents = []
        
        # Check significance for each constituent
        for col in columns:
            if not pd.isna(df_p.loc[df_p['GaugeName'] == gauge_name, col].iloc[0]):
                p_value = df_p.loc[df_p['GaugeName'] == gauge_name, col].iloc[0]
                corr_value = df_dep.loc[df_dep['GaugeName'] == gauge_name, col].iloc[0]
                
                if p_value < 0.1:  # Significant
                    significant_constituents.append({
                        'constituent': col,
                        'correlation_value': corr_value,
                        'p_value': p_value
                    })
        
        # Only include gauges with at least one significant constituent
        if significant_constituents:
            # Get primary and secondary constituents for this gauge
            temp_df = df_max_list[gauge_name]
            primary_constituent = temp_df['Prime'].iat[0] if 'Prime' in temp_df.columns else 'Unknown'
            secondary_constituent = temp_df['Secondary'].iat[0] if 'Secondary' in temp_df.columns and not pd.isna(temp_df['Secondary'].iat[0]) else 'None'
        
            significance_data.append({
                'GaugeName': gauge_name,
                'Latitude': lat,
                'Longitude': lon,
                'Years_of_Data': years,
                'Primary_Constituent': primary_constituent,
                'Secondary_Constituent': secondary_constituent,
                'Num_Significant': len(significant_constituents),
                'Significant_Constituents': ', '.join([f"{sc['constituent']}(r={sc['correlation_value']:.3f}, p={sc['p_value']:.3f})" for sc in significant_constituents])
            })

    # Create DataFrame and save to CSV
    if significance_data:
        significance_df = pd.DataFrame(significance_data)
        significance_df = significance_df.sort_values(['Num_Significant', 'GaugeName'], ascending=[False, True])
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "reports", f"gauge_significance_report_{method_name.lower()}.csv")
        significance_df.to_csv(csv_path, index=False)
        
        return significance_df
    
    return pd.DataFrame()

def compare_correlation_methods(kendall_dep, kendall_p, pearson_dep, pearson_p, columns, output_dir):
    """
    Create comprehensive comparison between Kendall tau and Pearson correlations
    """
    print("\nGenerating comparison analysis between Kendall tau and Pearson correlations...")
    
    # Create comparison plots for each constituent
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    comparison_data = []
    
    for i, col in enumerate(columns):
        # Get valid data for both methods
        kendall_valid = kendall_dep[col].notna() & kendall_p[col].notna()
        pearson_valid = pearson_dep[col].notna() & pearson_p[col].notna()
        
        # Find common stations
        common_stations = kendall_valid & pearson_valid
        
        if common_stations.sum() > 0:
            kendall_vals = kendall_dep.loc[common_stations, col]
            pearson_vals = pearson_dep.loc[common_stations, col]
            
            # Scatter plot
            axes[i].scatter(kendall_vals, pearson_vals, alpha=0.6, s=30)
            axes[i].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)  # diagonal line
            axes[i].set_xlabel('Kendall tau')
            axes[i].set_ylabel('Pearson r')
            axes[i].set_title(f'{col} Correlations\n(n={common_stations.sum()})')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(-1, 1)
            axes[i].set_ylim(-1, 1)
            
            # Calculate comparison statistics
            if len(kendall_vals) > 1:
                corr_between_methods = stats.pearsonr(kendall_vals, pearson_vals)[0]
                mean_diff = (pearson_vals - kendall_vals).mean()
                std_diff = (pearson_vals - kendall_vals).std()
                
                comparison_data.append({
                    'Constituent': col,
                    'N_Stations': common_stations.sum(),
                    'Kendall_Mean': kendall_vals.mean(),
                    'Pearson_Mean': pearson_vals.mean(),
                    'Kendall_Std': kendall_vals.std(),
                    'Pearson_Std': pearson_vals.std(),
                    'Correlation_Between_Methods': corr_between_methods,
                    'Mean_Difference_P_minus_K': mean_diff,
                    'Std_Difference': std_diff,
                    'Kendall_Significant': (kendall_p.loc[common_stations, col] < 0.1).sum(),
                    'Pearson_Significant': (pearson_p.loc[common_stations, col] < 0.1).sum()
                })
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_method_comparison_scatter.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison statistics DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "correlation_methods_comparison_stats.csv"), index=False)
    
    # Create difference distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        # Get valid data for both methods
        kendall_valid = kendall_dep[col].notna() & kendall_p[col].notna()
        pearson_valid = pearson_dep[col].notna() & pearson_p[col].notna()
        common_stations = kendall_valid & pearson_valid
        
        if common_stations.sum() > 0:
            kendall_vals = kendall_dep.loc[common_stations, col]
            pearson_vals = pearson_dep.loc[common_stations, col]
            differences = pearson_vals - kendall_vals
            
            axes[i].hist(differences, bins=20, alpha=0.7, edgecolor='black')
            axes[i].axvline(differences.mean(), color='red', linestyle='--', 
                           label=f'Mean: {differences.mean():.3f}')
            axes[i].set_xlabel('Pearson - Kendall')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{col} Differences\n(n={common_stations.sum()})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_differences_histogram.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create significance agreement analysis
    significance_agreement = []
    
    for col in columns:
        kendall_valid = kendall_dep[col].notna() & kendall_p[col].notna()
        pearson_valid = pearson_dep[col].notna() & pearson_p[col].notna()
        common_stations = kendall_valid & pearson_valid
        
        if common_stations.sum() > 0:
            kendall_sig = kendall_p.loc[common_stations, col] < 0.1
            pearson_sig = pearson_p.loc[common_stations, col] < 0.1
            
            both_sig = (kendall_sig & pearson_sig).sum()
            kendall_only = (kendall_sig & ~pearson_sig).sum()
            pearson_only = (~kendall_sig & pearson_sig).sum()
            neither_sig = (~kendall_sig & ~pearson_sig).sum()
            
            agreement_rate = (both_sig + neither_sig) / common_stations.sum()
            
            significance_agreement.append({
                'Constituent': col,
                'N_Stations': common_stations.sum(),
                'Both_Significant': both_sig,
                'Kendall_Only_Significant': kendall_only,
                'Pearson_Only_Significant': pearson_only,
                'Neither_Significant': neither_sig,
                'Agreement_Rate': agreement_rate
            })
    
    significance_df = pd.DataFrame(significance_agreement)
    significance_df.to_csv(os.path.join(output_dir, "significance_agreement_analysis.csv"), index=False)
    
    return comparison_df, significance_df

# Main analysis starts here
print("Enhanced Tidal Correlation Analysis - Kendall tau vs Pearson")
print("="*70)

# List files that we have in the correct folder
files = glob.glob(os.path.join(DATA_FOLDER, "*.mat"))
print(f"Found {len(files)} MAT files in {DATA_FOLDER}")

locations = []
# Add the latitude and longitude of stations to the list
for idx in range(len(files)):
    file = files[idx]
    try:
        data = loadmat(file)
        lat = float(np.squeeze(data['lat']))
        lon = float(np.squeeze(data['lon']))
        station_name = os.path.basename(file).split(".")[0]
        locations.append((lat, lon, station_name))
    except Exception as e:
        print(f"Error processing file {os.path.basename(file)}: {e}")
        continue

print(f"Successfully processed {len(locations)} station locations")

# Convert to DataFrame
df = pd.DataFrame(locations, columns=["lat", "lon", "GaugeName"])

# Convert to GeoDataFrame for reference map
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
)

# Create reference map with Robinson projection
print("Creating reference station location map with Robinson projection...")
fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.Robinson()})
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
ax.set_global()

# Plot station locations
ax.scatter(df['lon'], df['lat'], s=50, color='red', alpha=0.8, 
           transform=ccrs.PlateCarree(), edgecolor='black', linewidth=0.3)

ax.set_title("Gauge Locations - Robinson Projection", fontsize=16, fontweight="bold", pad=20)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

reference_map_path = os.path.join(OUTPUT_BASE, "gaugeLocation_robinson.png")
fig.savefig(reference_map_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# Load all the data M2 N2 S2, O1, ...
start_origin = 719529

df_list = {}
df_max_list = {}

# Debugging counters
total_files = len(files)
stations_with_data = 0
stations_after_date_filter = 0
stations_with_valid_constituents = 0
stations_final = 0

print(f"\nDEBUGGING STATION FILTERING:")
print(f"Starting with {total_files} files")

columns = ['M2', 'N2', 'S2', 'O1', 'K1', 'MK3', 'MS4']

for file in files:
    data = loadmat(file)
    station_name = os.path.basename(file).split(".")[0]
    cons_df = pd.DataFrame(data['Cons'], columns=columns)

    T_hat_flat = data['t_HAT'].flatten()
    cons_df['Date'] = pd.to_datetime(T_hat_flat - start_origin, unit='D', origin='unix').floor('D')
    
    # Count stations with any data
    if len(cons_df) > 0:
        stations_with_data += 1
    
    if len(cons_df) > 0:
        stations_after_date_filter += 1
        cons_df = cons_df.dropna().reset_index(drop=True)
        cols_no_allna = cons_df.iloc[:, :-1].dropna(axis=1, how='all')

        if not cols_no_allna.empty:
            stations_with_valid_constituents += 1
            # Get primary AND secondary constituents
            mean_amplitudes = cols_no_allna.mean(skipna=True).sort_values(ascending=False)
            primary = mean_amplitudes.index[0]
            secondary = mean_amplitudes.index[1] if len(mean_amplitudes) > 1 else None
        else:
            primary = None
            secondary = None
            cons_df = []
    else:
        primary = None
        secondary = None
        cons_df = []
        
    if len(cons_df) > 0:
        df_list[station_name] = cons_df.copy()
        years = cons_df.Date.dt.year.unique()
        
        # Check if station has minimum years BEFORE creating df_max_list
        if len(years) >= MIN_YEARS:
            stations_final += 1
            
        prim_max = cons_df.set_index("Date").resample("YE").max().reset_index()[['Date', primary]]
        max_cons_df = pd.DataFrame(index=prim_max.index, columns=list(cons_df.columns))
        max_cons_df['Date'] = prim_max.Date.dt.year
        max_cons_df[primary] = prim_max[primary] 
        cols_to_assign = max_cons_df.columns[:-1]
    
        max_cons_df[cols_to_assign] = max_cons_df[cols_to_assign].apply(pd.to_numeric, errors="coerce").astype(np.float64)   
        for row in range(len(max_cons_df)):
            primary_value = max_cons_df.iloc[row][primary]
            year = max_cons_df.iloc[row]["Date"]
            if not pd.isna(primary_value):
                mask = (cons_df[primary] == primary_value) & (cons_df["Date"].dt.year == year)
                if mask.any():
                    mask_index = cons_df[mask].index[0]
                    window_max = cons_df.iloc[mask_index-1:mask_index+2][cols_to_assign].max().astype(np.float64)
                    max_cons_df.loc[row, cols_to_assign] = window_max
        
        max_cons_df['Prime'] = primary
        max_cons_df['Secondary'] = secondary
        max_cons_df['year'] = len(years)
        df_max_list[station_name] = max_cons_df

print(f"Stations with any data: {stations_with_data}")
print(f"Stations after 1950-2019 filter: {stations_after_date_filter}")
print(f"Stations with valid constituents: {stations_with_valid_constituents}")
print(f"Stations with >= {MIN_YEARS} years: {stations_final}")
print(f"Stations lost: {total_files - stations_final}")

# Show breakdown of years available
if df_max_list:
    years_available = [df_max_list[key]['year'].iat[0] for key in df_max_list.keys()]
    years_df = pd.Series(years_available)
    print(f"\nYEARS OF DATA DISTRIBUTION:")
    print(f"Mean years: {years_df.mean():.1f}")
    print(f"Min years: {years_df.min()}")
    print(f"Max years: {years_df.max()}")
    print(f"Stations with < {MIN_YEARS} years: {(years_df < MIN_YEARS).sum()}")
    print(f"Stations with >= {MIN_YEARS} years: {(years_df >= MIN_YEARS).sum()}")

# Process for both correlation methods
methods = [
    {'name': 'Kendall', 'dir': KENDALL_DIR, 'method': 'kendall'},
    {'name': 'Pearson', 'dir': PEARSON_DIR, 'method': 'pearson'}
]

correlation_results = {}

for method_info in methods:
    method_name = method_info['name']
    method_dir = method_info['dir']
    correlation_method = method_info['method']
    
    print(f"\n{'='*50}")
    print(f"PROCESSING {method_name.upper()} CORRELATIONS")
    print(f"{'='*50}")
    
    # Create dependence dataframe
    dep_df = pd.DataFrame(locations, columns=["lat", "lon", "GaugeName"])
    for col in columns:
        dep_df[col] = 1.0
            
    dep_df_p = dep_df.copy()
    dep_df['years'] = 0

    dep_df = dep_df[dep_df['GaugeName'].isin(df_max_list.keys())]
    dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(df_max_list.keys())]

    # Calculate correlations using specified method
    for key in dep_df['GaugeName']:
        temp_df = df_max_list[key]
        temp_df = temp_df.dropna()

        primary = temp_df['Prime'].iat[0]
        secondary = temp_df['Secondary'].iat[0] if 'Secondary' in temp_df.columns and not pd.isna(temp_df['Secondary'].iat[0]) else None
        dep_df.loc[dep_df['GaugeName'] == key, "years"] = temp_df['year'].iat[0]
        
        # Calculate correlations and p-values
        correlations, p_values = calculate_correlations(temp_df, primary, secondary, columns, correlation_method)
        
        for col in columns:
            if col in correlations:
                dep_df.loc[dep_df['GaugeName'] == key, col] = correlations[col]
                dep_df_p.loc[dep_df_p['GaugeName'] == key, col] = p_values[col]

    # Filtering the rows with less than minimum years
    dep_df = dep_df[dep_df['years'] >= MIN_YEARS]
    dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(dep_df['GaugeName'])]

    print(f"Final dataset: {len(dep_df)} stations with >= {MIN_YEARS} years of data")

    # Store results for comparison
    correlation_results[method_name.lower()] = {
        'dep_df': dep_df.copy(),
        'dep_df_p': dep_df_p.copy()
    }

    # Generate summary report
    summary_lines = generate_summary_report(dep_df, dep_df_p, columns, df_max_list, method_name, method_dir)
    
    # Print summary to console
    for line in summary_lines:
        print(line)

    # Create significance report
    significance_df = create_significance_report(dep_df, dep_df_p, columns, df_max_list, method_name, method_dir)
    
    if not significance_df.empty:
        print(f"\n{method_name} significance report saved with {len(significance_df)} gauges")
        print(f"Summary of significant constituents per gauge:")
        print(significance_df['Num_Significant'].value_counts().sort_index())

    # Generate maps with Robinson projection and subplot labels
    print(f"\nGenerating {method_name} Robinson projection maps...")
    subplot_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    for i, col in enumerate(columns):
        subplot_label = subplot_letters[i] if i < len(subplot_letters) else f"({i+1})"
        print(f"Generating map {subplot_label}) for constituent {col}")
        plot_gauges(dep_df_p, dep_df, col, "enhanced_1950-2019", subplot_label, method_dir, method_name)

    print(f"{method_name} correlation analysis complete!")

# Perform comparison analysis
print(f"\n{'='*50}")
print("PERFORMING COMPARISON ANALYSIS")
print(f"{'='*50}")

kendall_results = correlation_results['kendall']
pearson_results = correlation_results['pearson']

comparison_stats, significance_agreement = compare_correlation_methods(
    kendall_results['dep_df'], kendall_results['dep_df_p'],
    pearson_results['dep_df'], pearson_results['dep_df_p'],
    columns, COMPARISON_DIR
)

print("Comparison analysis complete!")
print(f"Results saved in: {COMPARISON_DIR}")

# Print comparison summary
print(f"\nCOMPARISON SUMMARY:")
print("="*50)
for _, row in comparison_stats.iterrows():
    print(f"{row['Constituent']}:")
    print(f"  - Stations compared: {row['N_Stations']}")
    print(f"  - Correlation between methods: {row['Correlation_Between_Methods']:.3f}")
    print(f"  - Mean difference (P-K): {row['Mean_Difference_P_minus_K']:.3f}")
    print(f"  - Kendall significant: {row['Kendall_Significant']}")
    print(f"  - Pearson significant: {row['Pearson_Significant']}")

print(f"\nSignificance Agreement:")
print("="*50)
for _, row in significance_agreement.iterrows():
    print(f"{row['Constituent']}: {row['Agreement_Rate']:.3f} agreement rate")

print(f"\nAnalysis complete! All results saved in '{OUTPUT_BASE}' directory.")
print(f"Directory structure:")
print(f"  - {KENDALL_DIR}/")
print(f"    - maps/ (Kendall tau correlation maps)")
print(f"    - reports/ (Kendall tau analysis reports)")
print(f"  - {PEARSON_DIR}/")
print(f"    - maps/ (Pearson correlation maps)")
print(f"    - reports/ (Pearson analysis reports)")
print(f"  - {COMPARISON_DIR}/")
print(f"    - comparison statistics and plots")
