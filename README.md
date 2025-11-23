# GlobeTideCorr

This repository contains code and analysis for the global tidal-constituent co-variability study. The project analyzes long-record tide-gauge observations to evaluate annual maxima of major tidal constituents, identify primary and secondary constituents, compute enhanced Kendall τ correlations, and generate global Robinson-projection visualizations. The analyses include historical (1950–1980), modern (1981–2019), full-record (1950–2019), ENSO-conditioned subsets, and global primary-constituent distributions.

## Cite

If you use the codes, data, ideas, or results from this project, please cite the following paper:

**Radfar, S., Taheri, P., & Moftakhari, H. Global evidence for coherent variability in major tidal constituents. Environmental Research Letters (2025).**

(DOI will be added upon publication)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

To run the code in this repository, you'll need the following dependencies installed:

### Python Dependencies
- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- SciPy
- GeoPandas
- Cartopy
- Contextily

Install packages with:
```bash
pip install numpy pandas matplotlib scipy geopandas cartopy contextily
```

## Usage

Each script file includes a descriptive header. Main scripts include:

- `1950_robs.py`: Robinson-projection correlation maps for 1950–1980.
- `1981_robs.py`: Robinson-projection correlation maps for 1981–2019.
- `2019_robs.py`: Full-record (1950–2019) enhanced correlation analysis.
- `ENSO_analysis.py`: ENSO-phase-conditioned tidal-constituent correlations.
- `ENSO_robs.py`: Combined 7×3 ENSO-conditioned Robinson visualization.
- `primary_plot.py`: Global primary constituent distribution map.
- `enhanced_tidal_correlation_analysis.py`: Enhanced Kendall τ calculations.

## File Structure
```
├── data/
│   ├── station_1.mat
│   ├── station_2.mat
│   └── .gitkeep
├── scripts/
│   ├── 1950_robs.py
│   ├── 1981_robs.py
│   ├── 2019_robs.py
│   ├── ENSO_analysis.py
│   ├── ENSO_robs.py
│   ├── enhanced_tidal_correlation_analysis.py
│   ├── primary_plot.py
│   └── .gitkeep
├── LICENSE
└── README.md
```

## Data

The `data/` folder contains processed `.mat` files of tidal constituent amplitudes for all global tide gauges used in this study. These include:

- Annual constituent amplitude time series (M2, N2, S2, O1, K1, MK3, MS4)
- Station metadata (lat, lon, gauge name)
- Matrices required for enhanced Kendall τ computations

Raw source datasets cannot be redistributed due to licensing restrictions.

## Results

Outputs include:

- Global correlation maps for historical, modern, and full periods
- ENSO-conditioned correlation maps
- Primary constituent global distribution maps
- Station-level significance tables
- Enhanced τ correlation distributions

## Contributing

Contributions are welcome. Submit issues or pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

Supported by the Center for Complex Hydrosystems Research at the University of Alabama and NOAA CIROH Cooperative Agreement NA22NWS4320003.

## Contact

For questions, contact **sradfar@ua.edu**.
