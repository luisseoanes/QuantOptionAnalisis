# Configuración del proyecto
import numpy as np

# Parámetros generales
TICKER = 'AAPL'
RISK_FREE_RATE = 0.045  # Tasa libre de riesgo (aproximada, 4.5%)

# Parámetros de preprocesamiento
MIN_VOLUME = 0
MIN_OPEN_INTEREST = 0
MIN_PRICE = 0.01

# Parámetros de densidad
SPLINE_SMOOTHING = 0.01  # Parámetro s para UnivariateSpline
N_STRIKES_GRID = 500     # Puntos en la grilla de strikes

# Parámetros de entropía
RENYI_ALPHAS = [0.5, 1.5, 2.0]

# Parámetros de Monte Carlo
N_SIMULATIONS = 10000
N_STRESS_SCENARIOS = 5

# Parámetros históricos
HISTORICAL_YEARS = 3

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
PLOTS_DIR = 'output/plots'
