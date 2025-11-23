# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu)
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified: Nov 22, 2025
#
# Description:
# This script evaluates enhanced co-variability among major tidal constituents
# (M2, N2, S2, O1, K1, MK3, MS4) for the modern period (1981–2019). For every
# tide gauge, annual maxima are derived, primary and secondary constituents are
# identified, and Kendall τ correlations are computed between the primary
# constituent and all remaining constituents. Significant and non-significant
# correlations are displayed using Robinson-projection global maps.
#
# Outputs:
# - Seven Robinson-projection maps (one per constituent), showing τ correlations
#   and significance (p < 0.1) across the global gauge network.
# - A global reference map of all station locations.
# - A gauge-level significance report summarizing the number of significant
#   constituent pairs and associated τ and p-values.
# - Diagnostic tables describing station filtering, years of record, and
#   primary–secondary constituent distributions.
#
# For methodological context and scientific interpretation, see:
# Radfar, S., Taheri, P., and Moftakhari, H. (2025). Global evidence for
# coherent variability in major tidal constituents. *Environmental Research
# Letters*.
#
# Disclaimer:
# This script is provided for research and educational use only and is supplied
# “as is” without warranty of any kind. The author assumes no responsibility
# for errors, omissions, or consequences arising from its use.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import contextily as ctx
import geopandas as gpd
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

# Define the correct path to your data folder
DATA_FOLDER = r"D:\All stations calculated 2019\Station data"

# Define minimum years parameter
MIN_YEARS = 20

def plot_gauges(df_p, df_dep, col, name, subplot_label, latmin=None, latmax=None):
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
        cb.set_label('Tau value', fontsize=14, fontweight='bold')
    
    # Add title and subplot label
    # ax.set_title(f"Enhanced: Primary/Secondary with {col}", fontsize=16, fontweight="bold", pad=20)
    
    # Add subplot label (a), b), c), etc.) at top-left (no box)
    ax.text(0.02, 0.98, f'{subplot_label})', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left')
    
    # Add legend
    # ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Add gridlines (no right labels to remove 60°N, 60°S from right side)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Save figure
    fig.savefig(f"enhanced_dep{col}_{name}_robinson.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

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
# ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
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

fig.savefig("gaugeLocation_robinson.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
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

for file in files:
    data = loadmat(file)
    station_name = os.path.basename(file).split(".")[0]
    columns = ['M2', 'N2', 'S2', 'O1', 'K1', 'MK3', 'MS4']
    cons_df = pd.DataFrame(data['Cons'], columns=columns)

    T_hat_flat = data['t_HAT'].flatten()
    cons_df['Date'] = pd.to_datetime(T_hat_flat - start_origin, unit='D', origin='unix').floor('D')
    
    # Count stations with any data
    if len(cons_df) > 0:
        stations_with_data += 1
    
    # Filter the years between 1981 to 2019
    cons_df = cons_df[(cons_df.Date.dt.year >= 1981)]
    
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
print(f"Stations after 1981-2019 filter: {stations_after_date_filter}")
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

# Create dependence dataframe
dep_df = pd.DataFrame(locations, columns=["lat", "lon", "GaugeName"])

for col in columns:
    dep_df[col] = 1.0
        
dep_df_p = dep_df.copy()
dep_df['years'] = 0

dep_df = dep_df[dep_df['GaugeName'].isin(df_max_list.keys())]
dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(df_max_list.keys())]

# Enhanced correlation calculation
for key in dep_df['GaugeName']:
    temp_df = df_max_list[key]
    temp_df = temp_df.dropna()

    primary = temp_df['Prime'].iat[0]
    secondary = temp_df['Secondary'].iat[0] if 'Secondary' in temp_df.columns and not pd.isna(temp_df['Secondary'].iat[0]) else None
    dep_df.loc[dep_df['GaugeName'] == key, "years"] = temp_df['year'].iat[0]
    
    # Enhanced correlation logic for each target constituent
    for col in columns:
        if col == primary:
            # Target constituent IS primary - correlate with secondary
            if secondary and secondary in temp_df.columns:
                tau, kendall_p = stats.kendalltau(temp_df[primary], temp_df[secondary])
                dep_df.loc[dep_df['GaugeName'] == key, col] = tau
                dep_df_p.loc[dep_df_p['GaugeName'] == key, col] = kendall_p
        else:
            # Target constituent is NOT primary - correlate primary with target
            if col in temp_df.columns:
                tau, kendall_p = stats.kendalltau(temp_df[primary], temp_df[col])
                dep_df.loc[dep_df['GaugeName'] == key, col] = tau
                dep_df_p.loc[dep_df_p['GaugeName'] == key, col] = kendall_p

# Filtering the rows with less than minimum years
dep_df = dep_df[dep_df['years'] >= MIN_YEARS]
dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(dep_df['GaugeName'])]

print(f"Final dataset: {len(dep_df)} stations with >= {MIN_YEARS} years of data")

# Generate enhanced correlation analysis summary report
print(f"\n" + "="*80)
print("ENHANCED CORRELATION ANALYSIS SUMMARY:")
print("="*80)

for target_constituent in columns:
    # Count stations for this constituent
    valid_corr = dep_df[target_constituent].notna()
    total_stations = valid_corr.sum()
    
    if total_stations > 0:
        # Get data for this constituent
        correlations_subset = dep_df[valid_corr]
        p_values_subset = dep_df_p[valid_corr]
        
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
        
        print(f"\n{target_constituent} CORRELATIONS ({total_stations} total stations):")
        print(f"  - When {target_constituent} is PRIMARY: {primary_is_target} stations ({target_constituent} vs secondary)")
        print(f"  - When {target_constituent} is NOT primary: {primary_not_target} stations (primary vs {target_constituent})")
        print(f"  - Significant correlations (p < 0.1): {significant} stations ({significant/total_stations*100:.1f}%)")
        
        if most_common_secondary:
            print(f"  - Most common secondary when {target_constituent} is primary: {most_common_secondary} ({most_common_count} stations)")

# Generate plots with Robinson projection and subplot labels
print(f"\nGenerating Robinson projection maps...")
subplot_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

for i, col in enumerate(columns):
    subplot_label = subplot_letters[i] if i < len(subplot_letters) else f"({i+1})"
    print(f"Generating map {subplot_label}) for constituent {col}")
    plot_gauges(dep_df_p, dep_df, col, "enhanced_1981-2019", subplot_label)

print("Enhanced correlation analysis complete!")
print(f"Generated {len(columns)} Robinson projection maps with subplot labels.")

# Create significance report for each gauge
print(f"\nCreating significance report...")
significance_data = []

for idx, row in dep_df.iterrows():
    gauge_name = row['GaugeName']
    lat = row['lat']
    lon = row['lon']
    years = row['years']
    
    significant_constituents = []
    
    # Check significance for each constituent
    for col in columns:
        if not pd.isna(dep_df_p.loc[dep_df_p['GaugeName'] == gauge_name, col].iloc[0]):
            p_value = dep_df_p.loc[dep_df_p['GaugeName'] == gauge_name, col].iloc[0]
            tau_value = dep_df.loc[dep_df['GaugeName'] == gauge_name, col].iloc[0]
            
            if p_value < 0.1:  # Significant
                significant_constituents.append({
                    'constituent': col,
                    'tau_value': tau_value,
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
            'Significant_Constituents': ', '.join([f"{sc['constituent']}(tau={sc['tau_value']:.3f}, p={sc['p_value']:.3f})" for sc in significant_constituents])
        })

# Create DataFrame and save to CSV
if significance_data:
    significance_df = pd.DataFrame(significance_data)
    significance_df = significance_df.sort_values(['Num_Significant', 'GaugeName'], ascending=[False, True])
    
    # Save to CSV
    significance_df.to_csv('gauge_significance_report_1981.csv', index=False)
    
    print(f"Significance report saved to 'gauge_significance_report.csv'")
    print(f"Report includes {len(significance_df)} gauges with at least one significant constituent")
    print(f"Excluded {len(dep_df) - len(significance_df)} gauges with no significant constituents")
    
    # Print summary statistics
    print(f"\nSummary of significant constituents per gauge:")
    print(significance_df['Num_Significant'].value_counts().sort_index())
else:
    print("No gauges with significant constituents found.")
