# STES-analysis
## Intro
This work analyzed and visualized the Techno-Economic characteristics of Seasonal Thermal Energy Storage in Local Multi-Energy Systems.The results show that STES is more economical in high-latitude cities with significant seasonal heat demand variations and its optimal capacities based on two key factors: 1) the heat and cooling load ratio in the total energy demand, and 2) the carbon price.

Codes for Paper “Techno-Economic Analysis of Seasonal Thermal Energy Storage in Local Multi-Energy Systems”

Authors: Boyuan Liu, Jiahao Ma, Zeyang Long, Xueyuan Cui, and Yi Wang

## Dependencies
- [python = 3.12.4]
- [Pyomo]
- [Pandas]
- [NumPy]
- [Matplotlib]
- [Jupyter Notebook](https://jupyter.org/)
## Structure
.
├── data/                   # Contains all program load data

│   └── all_data.json      # Raw data used by the programs

│

├── basic_results/         # Output directory for basic.ipynb

├── carbon_results/        # Output directory for carbon.ipynb  

├── scaling_results/       # Output directory for scaling.ipynb

│

├── optimal.py             # Main optimization model program

├── basic.ipynb            # Basic analysis notebook

├── carbon.ipynb           # Carbon-related analysis notebook  

├── data_reading.ipynb     # Notebook for load curve visualization (Generates Fig.2)

└── scaling.ipynb          # Scaling analysis notebook
