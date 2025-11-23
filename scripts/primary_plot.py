# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu)
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified: Nov 22, 2025
#
# Description:
# This script identifies the dominant (primary) tidal constituent at each
# long-record tide gauge by computing mean amplitudes of the major constituents
# (M2, N2, S2, O1, K1, MK3, MS4). Stations are analyzed individually using their
# full post-1981 records, and the primary constituent is mapped globally using
# color- and marker-coded symbols for clear visual distinction.
#
# Outputs:
# - Global map showing the spatial distribution of primary tidal constituents.
# - Summary table listing the number and percentage of stations dominated by
#   each constituent.
#
# For scientific context regarding global tidal-constituent behavior, see:
# Radfar, S., Taheri, P., and Moftakhari, H. (2025). Global evidence for
# coherent variability in major tidal constituents. *Environmental Research
# Letters*.
#
# Disclaimer:
# This script is intended for research and educational purposes only and is
# provided “as is” without warranty of any kind. The author assumes no
# responsibility for errors, omissions, or outcomes arising from its use.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import geopandas as gpd
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the correct path to your data folder
DATA_FOLDER = r"...\Station data"

# Define colors for each constituent
constituent_colors = {
    'M2': '#FF0000',    # Red
    'N2': '#FF7F00',    # Orange  
    'S2': '#00FF00',    # Yellow
    'O1': '#0000FF',    # Green
    'K1': '#4B0082',    # Blue
    'MK3': '#FFFF00',   # Indigo
    'MS4': '#8B00FF'    # Violet
}

# Define marker shapes for each constituent (for better distinction)
constituent_markers = {
    'M2': 'o',     # Circle
    'N2': 's',     # Square
    'S2': '^',     # Triangle up
    'O1': 'D',     # Diamond
    'K1': 'v',     # Triangle down
    'MK3': 'P',    # Plus (filled)
    'MS4': 'X'     # X (filled)
}

# List files that we have in the correct folder
files = glob.glob(os.path.join(DATA_FOLDER, "*.mat"))
print(f"Found {len(files)} MAT files in {DATA_FOLDER}")

# Store station information with primary constituent
station_data = []
start_origin = 719529
columns = ['M2', 'N2', 'S2', 'O1', 'K1', 'MK3', 'MS4']

# Process each station to find primary constituent
for file in files:
    try:
        data = loadmat(file)
        station_name = os.path.basename(file).split(".")[0]
        lat = float(np.squeeze(data['lat']))
        lon = float(np.squeeze(data['lon']))
        
        # Create DataFrame for constituents
        cons_df = pd.DataFrame(data['Cons'], columns=columns)
        
        # Add date information
        T_hat_flat = data['t_HAT'].flatten()
        cons_df['Date'] = pd.to_datetime(T_hat_flat - start_origin, unit='D', origin='unix').floor('D')
        
        # Filter the years between 1950 to 1980
        # cons_df = cons_df[(cons_df.Date.dt.year >= 1950) & (cons_df.Date.dt.year < 1981)]
        cons_df = cons_df[(cons_df.Date.dt.year >= 1981)]
        
        if len(cons_df) > 0:
            cons_df = cons_df.dropna().reset_index(drop=True)
            cols_no_allna = cons_df.iloc[:, :-1].dropna(axis=1, how='all')

            if not cols_no_allna.empty:
                # Find the primary constituent (highest average amplitude)
                primary = cols_no_allna.mean(skipna=True).idxmax()
                primary_amplitude = cols_no_allna.mean(skipna=True).max()
                
                station_data.append({
                    'lat': lat,
                    'lon': lon,
                    'station_name': station_name,
                    'primary_constituent': primary,
                    'primary_amplitude': primary_amplitude
                })
                
    except Exception as e:
        print(f"Error processing file {os.path.basename(file)}: {e}")
        continue

# Convert to DataFrame
df_primary = pd.DataFrame(station_data)
print(f"Successfully processed {len(df_primary)} stations")

# Print summary of primary constituents
print("\nPrimary Constituent Distribution:")
primary_counts = df_primary['primary_constituent'].value_counts()
for constituent, count in primary_counts.items():
    percentage = (count / len(df_primary)) * 100
    print(f"{constituent}: {count} stations ({percentage:.1f}%)")

# Create the map
fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)

# Plot each constituent with different colors and markers
for constituent in columns:
    constituent_data = df_primary[df_primary['primary_constituent'] == constituent]
    if len(constituent_data) > 0:
        scatter = ax.scatter(
            constituent_data['lon'],
            constituent_data['lat'],
            c=constituent_colors[constituent],
            marker=constituent_markers[constituent],
            s=120,
            edgecolor='black',
            linewidth=0.8,
            label=f'{constituent} ({len(constituent_data)} stations)',
            alpha=0.8,
            zorder=5
        )

# Customize the plot
ax.set_title('Primary Tidal Constituent by Station (1950-1980)', 
             fontsize=20, fontweight='bold', pad=20)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14, 'weight': 'bold'}
gl.ylabel_style = {'size': 14, 'weight': 'bold'}

# Create custom legend with both color and marker shape
from matplotlib.lines import Line2D

legend_elements = []
for constituent in columns:
    count = len(df_primary[df_primary['primary_constituent'] == constituent])
    if count > 0:
        legend_elements.append(
            Line2D([0], [0], 
                   marker=constituent_markers[constituent], 
                   color='w', 
                   markerfacecolor=constituent_colors[constituent],
                   markersize=12,
                   markeredgecolor='black',
                   markeredgewidth=0.8,
                   label=f'{constituent} ({count} stations)')
        )

# Add legend
legend = ax.legend(handles=legend_elements,
                  loc='lower left',
                  bbox_to_anchor=(0.02, 0.02),
                  fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  ncol=1,
                  title='Primary Tidal Constituent',
                  title_fontsize=14)

# Set extent with latitude limits (exclude polar regions)
ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
# Change the latitude limits (-60, 80) to whatever range you want

# Save the figure
plt.tight_layout()
plt.savefig("primary_constituents_1981.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

# Also create a summary table
print(f"\nSummary Table:")
print(f"{'Constituent':<12} {'Count':<8} {'Percentage':<12} {'Color':<10} {'Marker'}")
print("-" * 55)
for constituent in columns:
    count = len(df_primary[df_primary['primary_constituent'] == constituent])
    if count > 0:
        percentage = (count / len(df_primary)) * 100
        print(f"{constituent:<12} {count:<8} {percentage:<11.1f}% {constituent_colors[constituent]:<10} {constituent_markers[constituent]}")
