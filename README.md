# Options Analysis and Entropy of Implied Distributions - AAPL

## 📋 Project Description

This project implements a complete system to analyze Apple (AAPL) options, reconstruct the **risk-neutral density (RND)** implied in market prices, calculate measures of **entropy** (Shannon and Rényi), and simulate **stress scenarios** via Monte Carlo.

### Objectives

1. **Reconstruct the risk-neutral density** using:

   * Breeden-Litzenberger (model-free)
   * Maximum Entropy (MaxEnt)

2. **Measure uncertainty** with:

   * Shannon entropy
   * Rényi entropy (α = 0.5, 1.5, 2.0)
   * KL divergence between distributions

3. **Simulate stress scenarios** with models:

   * Black-Scholes (Geometric Brownian Motion)
   * Heston (stochastic volatility)
   * Merton Jump-Diffusion (jumps)

4. **Estimate risk premia** by comparing implied vs historical densities

---

## 🏗️ Project Structure

```
project_root/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup_project.py             # Setup script
├── main.py                      # Main executable script
│
├── src/                         # Project modules
│   ├── __init__.py
│   ├── config.py                # Configuration
│   ├── data_loader.py           # Data download
│   ├── preprocessing.py         # Data cleaning
│   ├── implied_vol.py           # Implied volatility calculation
│   ├── density_reconstruction.py # RND reconstruction
│   ├── entropy_calculator.py    # Entropy calculations
│   └── monte_carlo.py           # Monte Carlo simulations
│
├── data/                        # Downloaded data
│   ├── AAPL_options.csv
│   ├── AAPL_options_clean.csv
│   ├── AAPL_options_with_iv.csv
│   └── AAPL_historical.csv
│
└── output/                      # Results
    ├── rnd_densities.pkl
    ├── entropy_by_maturity.csv
    ├── monte_carlo_summary.csv
    ├── executive_summary.txt
    └── plots/                   # Charts
        ├── volatility_smile.png
        ├── rnd_densities.png
        ├── entropy_analysis.png
        └── mc_distributions.png
```

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.9 or higher
* pip (Python package manager)

### Step 1: Download the project

Save all project files in a folder.

### Step 2: Create project structure

```bash
python setup_project.py
```

This script creates all necessary folders and configuration files.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies:**

* `numpy` - Numerical computations
* `pandas` - Data manipulation
* `scipy` - Scientific functions & optimization
* `matplotlib`, `seaborn` - Visualization
* `yfinance` - Financial data download
* `scikit-learn` - Density estimation (KDE)

---

## 💻 System Usage

### Full Execution (Recommended)

Run the entire pipeline end-to-end:

```bash
python main.py
```

This will automatically perform:

1. Download of options and historical data
2. Preprocessing and cleaning
3. Implied volatility calculations
4. Density reconstruction (RND and physical)
5. Entropy calculations
6. Monte Carlo simulations
7. Executive summary generation

**Estimated runtime:** 3-5 minutes (depending on internet speed and data size)

### Command Line Options

```bash
# Use existing data (skip downloads)
python main.py --skip-download

# Skip plots (calculations only)
python main.py --skip-plots

# Combine options
python main.py --skip-download --skip-plots
```

### Modular Execution (Step-by-Step)

Run individual modules if needed:

```bash
# 1. Download data
python src/data_loader.py

# 2. Preprocess
python src/preprocessing.py

# 3. Calculate implied volatilities
python src/implied_vol.py

# 4. Reconstruct densities
python src/density_reconstruction.py

# 5. Compute entropies
python src/entropy_calculator.py

# 6. Monte Carlo simulation
python src/monte_carlo.py
```

---

## 📊 Outputs

### CSV Files (output/)

| File                       | Description                         |
| -------------------------- | ----------------------------------- |
| `entropy_by_maturity.csv`  | Shannon & Rényi entropy by maturity |
| `entropy_risk_premium.csv` | Entropy risk premia                 |
| `monte_carlo_summary.csv`  | Monte Carlo scenario statistics     |
| `scenario_entropies.csv`   | Entropies of simulated scenarios    |
| `rnd_summary.csv`          | RND density summary                 |

### Pickle Files (output/)

Pickled Python objects for later analysis:

```python
import pickle

# Load RND densities
with open('output/rnd_densities.pkl', 'rb') as f:
    densities = pickle.load(f)

# Load Monte Carlo results
with open('output/monte_carlo_results.pkl', 'rb') as f:
    mc_results = pickle.load(f)
```

### Charts (output/plots/)

1. **volatility_smile.png** - Volatility smile per maturity
2. **iv_by_strike.png** - IV vs strike (nearest maturity)
3. **rnd_densities.png** - Reconstructed risk-neutral densities
4. **rnd_cdf.png** - Cumulative distribution functions
5. **density_physical.png** - Historical density estimate
6. **entropy_analysis.png** - Full entropy analysis
7. **entropy_risk_premium.png** - Risk premia
8. **mc_distributions.png** - Monte Carlo scenario distributions
9. **mc_statistics.png** - Scenario statistics comparison
10. **mc_risk_metrics.png** - Risk metrics (VaR, ES)

---

## 🔬 Methodology

### 1. Risk-Neutral Density Reconstruction

#### Breeden-Litzenberger Method

Uses the second derivative of call prices w.r.t. strike:

```
q(K) = exp(rT) × ∂²C/∂K²
```

**Implementation:**

* Smoothed cubic splines interpolation
* Finite differences for numerical derivatives
* Normalization to integrate to 1

#### Maximum Entropy Method (MaxEnt)

Maximizes entropy subject to option price constraints:

```
max H = -∫ q(x) log q(x) dx
s.t. theoretical prices = observed prices
```

**Implementation:**

* Lagrange multiplier optimization
* Constraints based on option moments

### 2. Entropy Calculations

#### Shannon Entropy

```
H = -∫ p(x) log p(x) dx
```

#### Rényi Entropy

```
H_α = (1/(1-α)) × log(∫ p(x)^α dx)
```

#### KL Divergence

```
KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
```

### 3. Monte Carlo Simulation

#### Black-Scholes

```
S(T) = S₀ × exp((μ - 0.5σ²)T + σ√T×Z)
```

#### Heston (Stochastic Volatility)

```
dS = μS dt + √v S dW₁
dv = κ(θ - v)dt + σᵥ√v dW₂
Corr(dW₁, dW₂) = ρ
```

#### Merton Jump-Diffusion

```
dS = μS dt + σS dW + (J-1)S dN
where N ~ Poisson(λ), J ~ LogNormal(μⱼ, σⱼ²)
```

---

## ⚙️ Advanced Configuration

### Modify Parameters

Edit `src/config.py`:

```python
# Risk-free rate
RISK_FREE_RATE = 0.045  # 4.5%

# Spline smoothing
SPLINE_SMOOTHING = 0.01

# Monte Carlo simulations
N_SIMULATIONS = 10000

# Historical years
HISTORICAL_YEARS = 3
```

### Change Asset

To analyze another asset (e.g., MSFT):

1. Edit `main.py`:

```python
pipeline = OptionsAnalysisPipeline(
    ticker='MSFT',  # Change here
    skip_download=skip_download,
    skip_plots=skip_plots
)
```

2. Or adjust individual modules to use the new ticker.

---

## 🐛 Troubleshooting

### Error: "No options available"

**Cause:** Yahoo Finance may not have options data for the requested date.
**Solution:**

* Check if the market is open
* Use `--skip-download` if you already have data
* Try a highly liquid ticker (SPY, QQQ)

### Error: "Insufficient calls for reconstruction"

**Cause:** Few liquid options for a specific maturity.
**Solution:**

* Normal for long-dated maturities
* The system automatically skips these
* Only maturities with sufficient data are processed

### Error: "Module not found"

**Cause:** Missing dependencies.
**Solution:**

```bash
pip install -r requirements.txt --upgrade
```

### scipy/numpy Warnings

**Cause:** Normal during numerical optimizations.
**Solution:** Already suppressed with `warnings.filterwarnings('ignore')`

---

## 📈 Results Interpretation

### Shannon Entropy

* **High entropy (> 4.0 nats):** Higher market uncertainty
* **Low entropy (< 3.0 nats):** More concentrated, lower uncertainty

### KL Divergence

* **KL > 0.5:** Large difference between market expectations and historical distribution
* **KL ≈ 0:** Market aligned with historical behavior

### Risk Premia

```
Risk Premium = KL(RND || Physical)
```

Represents the "extra" the market charges for uncertainty vs historical observations.

---

## 📚 References

1. **Breeden, D. T., & Litzenberger, R. H. (1978).** "Prices of State-Contingent Claims Implicit in Option Prices." *Journal of Business*, 51(4), 621-651.

2. **Jackwerth, J. C. (2004).** "Option-Implied Risk-Neutral Distributions and Risk Aversion." *Research Foundation of AIMR*.

3. **Buchen, P. W., & Kelly, M. (1996).** "The Maximum Entropy Distribution of an Asset Inferred from Option Prices." *Journal of Financial and Quantitative Analysis*, 31(1), 143-159.

4. **Heston, S. L. (1993).** "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2), 327-343.

5. **Merton, R. C. (1976).** "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.

---

## 📝 Additional Notes

### Limitations

* **Real-time data:** yfinance may have a 15-20 min delay
* **American options:** Model assumes European (reasonable for OTM strikes)
* **Microstructure:** Bid-ask spreads not explicitly modeled (mid-price used)

### Possible Extensions

1. **Model calibration:** Calibrate Heston/Merton to observed prices
2. **Term structure:** Analyze volatility term structure
3. **Greeks:** Compute option sensitivities
4. **Backtesting:** Validate predictions with future data
5. **Machine Learning:** Forecast entropy shifts

---

## 👤 Author & License

**Academic Project**
Developed for quantitative financial market analysis.

**Usage:** Educational and research purposes only.

**Disclaimer:** This code is for educational purposes only. It does not constitute financial advice or investment recommendations.

---

## 🤝 Contributions

To report bugs or suggest improvements, please include:

1. Python version used
2. Full error message
3. Steps to reproduce the problem

---

**Ready to analyze! Run `python main.py` to get started.**

---

Do you want me to also prepare a **short executive summary in English** (like a one-page brief for non-technical readers), or just keep this full documentation?
