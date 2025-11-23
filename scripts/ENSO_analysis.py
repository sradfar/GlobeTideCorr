# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu)
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified: Nov 22, 2025
#
# Description:
# This script evaluates how the co-variability of major tidal constituents
# (M2, N2, S2, O1, K1, MK3, MS4) changes across ENSO phases. ONI data from NOAA
# are used to classify each year as El Niño, La Niña, or Neutral. Annual
# constituent maxima are extracted for each tide gauge, primary and secondary
# constituents are identified, and Kendall τ correlations are computed for each
# ENSO subset (with ≥5 years required per phase).
#
# Outputs:
# - ENSO-specific global correlation maps (El Niño, La Niña, Neutral).
# - Station-level summaries of significant correlations (p < 0.1).
# - Phase-wise comparison tables of mean τ and significance frequency.
# - Diagnostic counts of station eligibility across ENSO categories.
#
# For methodological details and scientific context, see:
# Radfar, S., Taheri, P., and Moftakhari, H. (2025). Global evidence for
# coherent variability in major tidal constituents. *Environmental Research
# Letters*.
#
# Disclaimer:
# This script is provided for research and educational purposes only and is
# supplied “as is” without warranty of any kind. The author assumes no
# responsibility for errors, omissions, or outcomes arising from its use.
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
import requests
from io import StringIO

# Define the correct path to your data folder
DATA_FOLDER = r"...\Station data"

# Define minimum years parameter
MIN_YEARS = 20

def load_oni_data():
    """
    Load Oceanic Niño Index (ONI) data and classify years as El Niño, La Niña, or Neutral
    """
    print("Loading ONI data from NOAA...")
    
    # Try to load from NOAA Climate Prediction Center
    url = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
    
    try:
        response = requests.get(url)
        content = response.text
        
        # Split into lines and filter out comments/headers
        lines = content.strip().split('\n')
        data_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('YR') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    try:
                        year = int(parts[0])
                        month = int(parts[1]) 
                        
                        # The NOAA format is: YR MON TOTAL ANOM CLIM
                        # We want the anomaly value, which is parts[4] (5th column)
                        oni_value = float(parts[4])  # This is the actual anomaly
                        data_lines.append([year, month, oni_value])
                    except (ValueError, IndexError):
                        continue
        
        # Create DataFrame
        oni_df = pd.DataFrame(data_lines, columns=['year', 'month', 'oni'])
        
        # Calculate DJF (December-January-February) seasonal averages
        yearly_classifications = {}
        
        years = sorted(oni_df['year'].unique())
        
        for year in years:
            # Get DJF months for this year
            # December of previous year, January and February of current year
            djf_months = []
            
            # December of previous year
            dec_data = oni_df[(oni_df['year'] == year-1) & (oni_df['month'] == 12)]
            if not dec_data.empty:
                djf_months.append(dec_data['oni'].iloc[0])
            
            # January of current year
            jan_data = oni_df[(oni_df['year'] == year) & (oni_df['month'] == 1)]
            if not jan_data.empty:
                djf_months.append(jan_data['oni'].iloc[0])
                
            # February of current year  
            feb_data = oni_df[(oni_df['year'] == year) & (oni_df['month'] == 2)]
            if not feb_data.empty:
                djf_months.append(feb_data['oni'].iloc[0])
            
            # Need at least 2 months to calculate average
            if len(djf_months) >= 2:
                djf_avg = np.mean(djf_months)
                
                # Classify based on NOAA criteria
                if djf_avg >= 0.5:
                    classification = 'El Niño'
                elif djf_avg <= -0.5:
                    classification = 'La Niña'
                else:
                    classification = 'Neutral'
                
                yearly_classifications[year] = {
                    'djf_oni': djf_avg,
                    'enso_class': classification
                }
        
        # Print summary
        if yearly_classifications:
            years_range = f"{min(yearly_classifications.keys())}-{max(yearly_classifications.keys())}"
            print(f"Successfully loaded ONI data for {len(yearly_classifications)} years ({years_range})")
            
            # Count classifications
            counts = {'El Niño': 0, 'La Niña': 0, 'Neutral': 0}
            for data in yearly_classifications.values():
                counts[data['enso_class']] += 1
            
            total = len(yearly_classifications)
            print("ENSO Classification Summary:")
            for enso_type, count in counts.items():
                print(f"  {enso_type}: {count} years ({count/total*100:.1f}%)")
        
        return yearly_classifications
        
    except Exception as e:
        print(f"Failed to load ONI data from NOAA: {e}")
        print("Using backup historical ENSO classifications...")
        
        # Backup: Known historical ENSO events
        backup_classifications = {
            # Strong/Moderate El Niño years
            1957: {'djf_oni': 1.0, 'enso_class': 'El Niño'},
            1965: {'djf_oni': 1.2, 'enso_class': 'El Niño'},
            1972: {'djf_oni': 1.8, 'enso_class': 'El Niño'},
            1982: {'djf_oni': 2.1, 'enso_class': 'El Niño'},
            1987: {'djf_oni': 1.1, 'enso_class': 'El Niño'},
            1991: {'djf_oni': 1.4, 'enso_class': 'El Niño'},
            1994: {'djf_oni': 0.9, 'enso_class': 'El Niño'},
            1997: {'djf_oni': 2.3, 'enso_class': 'El Niño'},
            2002: {'djf_oni': 1.1, 'enso_class': 'El Niño'},
            2004: {'djf_oni': 0.7, 'enso_class': 'El Niño'},
            2009: {'djf_oni': 1.0, 'enso_class': 'El Niño'},
            2015: {'djf_oni': 2.5, 'enso_class': 'El Niño'},
            
            # Strong/Moderate La Niña years
            1955: {'djf_oni': -1.2, 'enso_class': 'La Niña'},
            1970: {'djf_oni': -0.8, 'enso_class': 'La Niña'},
            1973: {'djf_oni': -1.5, 'enso_class': 'La Niña'},
            1975: {'djf_oni': -1.6, 'enso_class': 'La Niña'},
            1988: {'djf_oni': -1.8, 'enso_class': 'La Niña'},
            1995: {'djf_oni': -0.6, 'enso_class': 'La Niña'},
            1998: {'djf_oni': -1.4, 'enso_class': 'La Niña'},
            1999: {'djf_oni': -1.5, 'enso_class': 'La Niña'},
            2007: {'djf_oni': -1.2, 'enso_class': 'La Niña'},
            2010: {'djf_oni': -1.4, 'enso_class': 'La Niña'},
        }
        
        # Fill in neutral years for a reasonable range
        full_classifications = {}
        for year in range(1950, 2021):
            if year in backup_classifications:
                full_classifications[year] = backup_classifications[year]
            else:
                full_classifications[year] = {'djf_oni': 0.0, 'enso_class': 'Neutral'}
        
        print(f"Using backup ENSO data for {len(full_classifications)} years (1950-2020)")
        
        # Count classifications
        counts = {'El Niño': 0, 'La Niña': 0, 'Neutral': 0}
        for data in full_classifications.values():
            counts[data['enso_class']] += 1
        
        total = len(full_classifications)
        print("Backup ENSO Classification Summary:")
        for enso_type, count in counts.items():
            print(f"  {enso_type}: {count} years ({count/total*100:.1f}%)")
        
        return full_classifications

def plot_gauges_enso(df_p, df_dep, col, name, enso_phase, latmin=None, latmax=None):
    """
    Modified plotting function for ENSO analysis
    """
    fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
    
    if (latmin is not None) or (latmax is not None):
        lonmin, lonmax, cur_latmin, cur_latmax = ax.get_extent(crs=ccrs.PlateCarree())
        if latmin is None: latmin = cur_latmin
        if latmax is None: latmax = cur_latmax
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    
    sig = df_p[col] < 0.1
    nonsig = ~sig
    
    ax.scatter(
        df_dep.loc[nonsig, 'lon'],
        df_dep.loc[nonsig, 'lat'],
        s=20,
        color='gray',
        alpha=1.0,
        label='p ≥ 0.1'
    )
    
    scatter = ax.scatter(
        df_dep.loc[sig, 'lon'],
        df_dep.loc[sig, 'lat'],
        c=df_dep.loc[sig, col],
        s=100,
        cmap='seismic',
        marker=marks_dic[col],
        edgecolor='black',
        linewidth=0.5,
        label='p < 0.1',
        vmin=-1,
        vmax=1
    )
    
    ax.set_title(f"ENSO {enso_phase}: {col} Correlations", fontsize=16, fontweight="bold")
    cb = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
    cb.set_label('Kendall τ', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}
    
    filename = f"enso_{enso_phase.lower().replace(' ', '_')}_{col}_{name}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# Load ONI data
oni_data = load_oni_data()

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

# Load all the data
start_origin = 719529
df_list = {}
df_max_list = {}

# Define major constituents
major_constituents = ['M2', 'S2', 'O1', 'K1']
columns = ['M2', 'N2', 'S2', 'O1', 'K1', 'MK3', 'MS4']

# Debugging counters
total_files = len(files)
stations_with_data = 0
stations_after_date_filter = 0
stations_with_major_constituents = 0
stations_with_valid_constituents = 0
stations_final = 0

print(f"\nDEBUGGING STATION FILTERING:")
print(f"Starting with {total_files} files")

for file in files:
    data = loadmat(file)
    station_name = os.path.basename(file).split(".")[0]
    cons_df = pd.DataFrame(data['Cons'], columns=columns)

    T_hat_flat = data['t_HAT'].flatten()
    cons_df['Date'] = pd.to_datetime(T_hat_flat - start_origin, unit='D', origin='unix').floor('D')
    
    # Count stations with any data
    if len(cons_df) > 0:
        stations_with_data += 1
    
    # Use all available years (no date filtering)
    if len(cons_df) > 0:
        stations_after_date_filter += 1
        
        # ENHANCED: Less aggressive NaN handling - only require major constituents
        cons_df = cons_df.dropna(subset=major_constituents)
        
        if len(cons_df) > 0:
            stations_with_major_constituents += 1
            
            # Remove columns that are entirely NaN
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
    else:
        primary = None
        secondary = None
        cons_df = []
        
    if len(cons_df) > 0:
        df_list[station_name] = cons_df.copy()
        years = cons_df.Date.dt.year.unique()
        
        # Check if station has minimum years
        if len(years) >= MIN_YEARS:
            stations_final += 1
            
        # Create annual maximum dataframe with ENSO classification
        prim_max = cons_df.set_index("Date").resample("YE").max().reset_index()[['Date', primary]]
        max_cons_df = pd.DataFrame(index=prim_max.index, columns=list(cons_df.columns) + ['enso_class'])
        max_cons_df['Date'] = prim_max.Date.dt.year
        max_cons_df[primary] = prim_max[primary]
        cols_to_assign = [col for col in cons_df.columns if col != 'Date']
    
        max_cons_df[cols_to_assign] = max_cons_df[cols_to_assign].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        
        for row in range(len(max_cons_df)):
            primary_value = max_cons_df.iloc[row][primary]
            year = int(max_cons_df.iloc[row]["Date"])
            
            # Add ENSO classification
            if oni_data and year in oni_data:
                max_cons_df.loc[row, 'enso_class'] = oni_data[year]['enso_class']
            else:
                max_cons_df.loc[row, 'enso_class'] = 'Unknown'
            
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
print(f"Stations after date filter: {stations_after_date_filter}")
print(f"Stations with major constituents present: {stations_with_major_constituents}")
print(f"Stations with valid constituents: {stations_with_valid_constituents}")
print(f"Stations with >= {MIN_YEARS} years: {stations_final}")
print(f"Stations lost: {total_files - stations_final}")

# ENSO-specific analysis
enso_phases = ['El Niño', 'La Niña', 'Neutral']
enso_results = {}

for enso_phase in enso_phases:
    print(f"\n" + "="*80)
    print(f"ANALYZING {enso_phase.upper()} YEARS")
    print("="*80)
    
    # Create dependence dataframe for this ENSO phase
    dep_df = pd.DataFrame(locations, columns=["lat", "lon", "GaugeName"])
    for col in columns:
        dep_df[col] = np.nan
    dep_df_p = dep_df.copy()
    dep_df['years'] = 0
    dep_df['enso_years'] = 0
    
    dep_df = dep_df[dep_df['GaugeName'].isin(df_max_list.keys())]
    dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(df_max_list.keys())]
    
    # Calculate correlations for this ENSO phase
    for key in dep_df['GaugeName']:
        temp_df = df_max_list[key]
        
        # Filter for specific ENSO phase
        enso_filtered = temp_df[temp_df['enso_class'] == enso_phase].dropna()
        
        if len(enso_filtered) >= 5:  # Minimum 5 years for correlation
            primary = temp_df['Prime'].iat[0]
            secondary = temp_df['Secondary'].iat[0] if 'Secondary' in temp_df.columns and not pd.isna(temp_df['Secondary'].iat[0]) else None
            
            dep_df.loc[dep_df['GaugeName'] == key, "years"] = temp_df['year'].iat[0]
            dep_df.loc[dep_df['GaugeName'] == key, "enso_years"] = len(enso_filtered)
            
            # Calculate correlations for each target constituent
            for col in columns:
                if col == primary:
                    # Primary vs secondary correlation
                    if secondary and secondary in enso_filtered.columns:
                        if enso_filtered[primary].notna().sum() >= 5 and enso_filtered[secondary].notna().sum() >= 5:
                            tau, kendall_p = stats.kendalltau(enso_filtered[primary], enso_filtered[secondary])
                            dep_df.loc[dep_df['GaugeName'] == key, col] = tau
                            dep_df_p.loc[dep_df_p['GaugeName'] == key, col] = kendall_p
                else:
                    # Primary vs target correlation
                    if col in enso_filtered.columns:
                        if enso_filtered[primary].notna().sum() >= 5 and enso_filtered[col].notna().sum() >= 5:
                            tau, kendall_p = stats.kendalltau(enso_filtered[primary], enso_filtered[col])
                            dep_df.loc[dep_df['GaugeName'] == key, col] = tau
                            dep_df_p.loc[dep_df_p['GaugeName'] == key, col] = kendall_p
    
    # Filter for minimum years
    dep_df = dep_df[dep_df['years'] >= MIN_YEARS]
    dep_df_p = dep_df_p[dep_df_p['GaugeName'].isin(dep_df['GaugeName'])]
    
    print(f"Stations analyzed for {enso_phase}: {len(dep_df)}")
    
    # Store results
    enso_results[enso_phase] = {'dep_df': dep_df.copy(), 'dep_df_p': dep_df_p.copy()}
    
    # Generate summary for this ENSO phase
    for target_constituent in columns:
        valid_corr = dep_df[target_constituent].notna()
        total_stations = valid_corr.sum()
        
        if total_stations > 0:
            p_values_subset = dep_df_p[valid_corr]
            significant = (p_values_subset[target_constituent] < 0.1).sum()
            # mean_tau = dep_df[valid_corr][target_constituent].mean()
            mean_tau = dep_df[valid_corr & (dep_df_p[target_constituent] < 0.1)][target_constituent].mean()
            
            print(f"{target_constituent}: {total_stations} stations, "
                  f"{significant} significant ({significant/total_stations*100:.1f}%), "
                  f"mean τ = {mean_tau:.3f}")

# Generate comparison plots and analysis
marks = ['o', 's', '^', 'D', 'v', 'P', 'X']
marks_dic = {col: marker for col, marker in zip(columns, marks)}

print(f"\n" + "="*80)
print("ENSO COMPARISON ANALYSIS")
print("="*80)

# Create plots for each ENSO phase
for enso_phase in enso_phases:
    if enso_phase in enso_results:
        dep_df = enso_results[enso_phase]['dep_df']
        dep_df_p = enso_results[enso_phase]['dep_df_p']
        
        for col in columns:
            if dep_df[col].notna().sum() > 0:
                plot_gauges_enso(dep_df_p, dep_df, col, "all_years", enso_phase)

# Compare correlations between ENSO phases
comparison_summary = {}
for col in columns:
    comparison_summary[col] = {}
    
    for enso_phase in enso_phases:
        if enso_phase in enso_results:
            dep_df = enso_results[enso_phase]['dep_df']
            dep_df_p = enso_results[enso_phase]['dep_df_p']
            
            valid_corr = dep_df[col].notna()
            if valid_corr.sum() > 0:
                significant_mask = valid_corr & (dep_df_p[col] < 0.1)
                mean_tau = dep_df[significant_mask][col].mean()
                significant = (dep_df_p[valid_corr][col] < 0.1).sum()
                total = valid_corr.sum()
                
                comparison_summary[col][enso_phase] = {
                    'mean_tau': mean_tau,
                    'significant': significant,
                    'total': total,
                    'sig_percent': significant/total*100 if total > 0 else 0
                }

# Print comparison summary
print("\nCONSTITUENT CORRELATION COMPARISON ACROSS ENSO PHASES:")
print("-" * 70)
print(f"{'Constituent':<12} {'ENSO Phase':<12} {'Stations':<10} {'Mean τ':<10} {'Significant':<12}")
print("-" * 70)

for col in columns:
    if col in comparison_summary:
        for enso_phase in enso_phases:
            if enso_phase in comparison_summary[col]:
                data = comparison_summary[col][enso_phase]
                print(f"{col:<12} {enso_phase:<12} {data['total']:<10} "
                      f"{data['mean_tau']:<10.3f} {data['significant']:<4} ({data['sig_percent']:<5.1f}%)")
        print("-" * 70)

print("\nENSO-enhanced correlation analysis complete!")
print(f"Generated maps for {len(columns)} constituents across {len(enso_phases)} ENSO phases.")
