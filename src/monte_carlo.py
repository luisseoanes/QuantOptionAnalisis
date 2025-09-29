"""
monte_carlo.py
Módulo para simulación Monte Carlo de escenarios de estrés.
Implementa modelos: Black-Scholes, Heston, y Merton Jump-Diffusion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """
    Simulador Monte Carlo para modelos de precios de activos.
    """
    
    def __init__(self, S0, r=0.045, n_simulations=10000, seed=42):
        """
        Inicializa el simulador.
        
        Args:
            S0: Precio inicial del activo
            r: Tasa libre de riesgo
            n_simulations: Número de simulaciones
            seed: Semilla aleatoria
        """
        self.S0 = S0
        self.r = r
        self.n_simulations = n_simulations
        self.seed = seed
        np.random.seed(seed)
        
        self.simulation_results = {}
    
    def black_scholes_simulation(self, T, sigma, n_steps=252):
        """
        Simula precios usando modelo Black-Scholes (Geometric Brownian Motion).
        
        dS = μ*S*dt + σ*S*dW
        S(T) = S0 * exp((μ - 0.5*σ²)*T + σ*sqrt(T)*Z)
        
        Args:
            T: Tiempo hasta vencimiento (años)
            sigma: Volatilidad anual
            n_steps: Número de pasos temporales
            
        Returns:
            Array con precios finales simulados
        """
        # Simulación directa del precio final
        Z = np.random.standard_normal(self.n_simulations)
        
        # Fórmula de Black-Scholes
        ST = self.S0 * np.exp((self.r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        return ST
    
    def heston_simulation(self, T, v0, kappa, theta, sigma_v, rho, n_steps=252):
        """
        Simula precios usando modelo de Heston (volatilidad estocástica).
        
        dS = μ*S*dt + sqrt(v)*S*dW1
        dv = κ(θ - v)*dt + σ_v*sqrt(v)*dW2
        donde Corr(dW1, dW2) = ρ
        
        Args:
            T: Tiempo hasta vencimiento (años)
            v0: Varianza inicial
            kappa: Velocidad de reversión a la media
            theta: Varianza de largo plazo
            sigma_v: Volatilidad de la varianza
            rho: Correlación entre movimientos de precio y varianza
            n_steps: Pasos temporales
            
        Returns:
            Array con precios finales simulados
        """
        dt = T / n_steps
        
        # Arrays para almacenar trayectorias
        S = np.zeros((self.n_simulations, n_steps + 1))
        v = np.zeros((self.n_simulations, n_steps + 1))
        
        # Condiciones iniciales
        S[:, 0] = self.S0
        v[:, 0] = v0
        
        # Simulación
        for t in range(1, n_steps + 1):
            # Generar choques correlacionados
            Z1 = np.random.standard_normal(self.n_simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(self.n_simulations)
            
            # Actualizar varianza (esquema de Euler con truncamiento)
            v_sqrt = np.sqrt(np.maximum(v[:, t-1], 0))
            v[:, t] = v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + sigma_v * v_sqrt * np.sqrt(dt) * Z2
            v[:, t] = np.maximum(v[:, t], 0)  # Truncar en cero
            
            # Actualizar precio
            v_sqrt_current = np.sqrt(np.maximum(v[:, t-1], 0))
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * v[:, t-1]) * dt + v_sqrt_current * np.sqrt(dt) * Z1)
        
        return S[:, -1]
    
    def merton_jump_diffusion(self, T, sigma, lambda_jump, mu_jump, sigma_jump, n_steps=252):
        """
        Simula precios usando modelo de Merton con saltos (Jump-Diffusion).
        
        dS = μ*S*dt + σ*S*dW + (J-1)*S*dN
        donde dN es proceso de Poisson con intensidad λ
        y J = exp(Y) con Y ~ N(μ_J, σ_J²)
        
        Args:
            T: Tiempo hasta vencimiento (años)
            sigma: Volatilidad difusión
            lambda_jump: Intensidad de saltos (saltos/año)
            mu_jump: Media del tamaño de salto (en log)
            sigma_jump: Std dev del tamaño de salto
            n_steps: Pasos temporales
            
        Returns:
            Array con precios finales simulados
        """
        dt = T / n_steps
        
        # Ajuste de drift por saltos
        k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
        drift_adj = self.r - 0.5 * sigma**2 - lambda_jump * k
        
        # Arrays
        S = np.zeros((self.n_simulations, n_steps + 1))
        S[:, 0] = self.S0
        
        # Simulación
        for t in range(1, n_steps + 1):
            # Componente difusión
            Z = np.random.standard_normal(self.n_simulations)
            diffusion = drift_adj * dt + sigma * np.sqrt(dt) * Z
            
            # Componente de saltos
            N_jumps = np.random.poisson(lambda_jump * dt, self.n_simulations)
            jump_sizes = np.zeros(self.n_simulations)
            
            for i in range(self.n_simulations):
                if N_jumps[i] > 0:
                    # Sumar todos los saltos en este período
                    jumps = np.random.normal(mu_jump, sigma_jump, N_jumps[i])
                    jump_sizes[i] = np.sum(jumps)
            
            # Actualizar precio
            S[:, t] = S[:, t-1] * np.exp(diffusion + jump_sizes)
        
        return S[:, -1]
    
    def run_stress_scenarios(self, T, base_params, stress_configs):
        """
        Ejecuta múltiples escenarios de estrés.
        
        Args:
            T: Tiempo hasta vencimiento (años)
            base_params: dict con parámetros base
            stress_configs: lista de dicts con configuraciones de estrés
            
        Returns:
            dict con resultados de todos los escenarios
        """
        print(f"\n{'='*60}")
        print(f"SIMULACIÓN MONTE CARLO - ESCENARIOS DE ESTRÉS")
        print(f"{'='*60}")
        print(f"\n📊 Configuración:")
        print(f"  • Precio inicial: ${self.S0:.2f}")
        print(f"  • Horizonte: {T:.4f} años ({T*365:.0f} días)")
        print(f"  • Simulaciones: {self.n_simulations:,}")
        print(f"  • Tasa libre de riesgo: {self.r:.2%}")
        
        results = {}
        
        # Escenario base: Black-Scholes
        print(f"\n{'─'*60}")
        print("Escenario BASE: Black-Scholes")
        print(f"  σ = {base_params['sigma']:.2%}")
        
        ST_base = self.black_scholes_simulation(T, base_params['sigma'])
        results['base_bs'] = self._analyze_simulation(ST_base, "Base BS")
        
        # Escenarios de estrés
        for idx, config in enumerate(stress_configs, 1):
            print(f"\n{'─'*60}")
            print(f"Escenario de ESTRÉS {idx}: {config['name']}")
            print(f"  Modelo: {config['model']}")
            
            if config['model'] == 'black_scholes':
                print(f"  σ = {config['sigma']:.2%}")
                ST = self.black_scholes_simulation(T, config['sigma'])
                
            elif config['model'] == 'heston':
                print(f"  v0 = {config['v0']:.4f}, κ = {config['kappa']:.2f}")
                print(f"  θ = {config['theta']:.4f}, σ_v = {config['sigma_v']:.2f}")
                print(f"  ρ = {config['rho']:.2f}")
                ST = self.heston_simulation(
                    T, config['v0'], config['kappa'], config['theta'],
                    config['sigma_v'], config['rho']
                )
                
            elif config['model'] == 'merton_jump':
                print(f"  σ = {config['sigma']:.2%}, λ = {config['lambda_jump']:.2f}")
                print(f"  μ_J = {config['mu_jump']:.4f}, σ_J = {config['sigma_jump']:.4f}")
                ST = self.merton_jump_diffusion(
                    T, config['sigma'], config['lambda_jump'],
                    config['mu_jump'], config['sigma_jump']
                )
            
            results[f"stress_{idx}_{config['model']}"] = self._analyze_simulation(
                ST, config['name']
            )
        
        self.simulation_results = results
        
        print(f"\n{'='*60}")
        print(f"✅ SIMULACIÓN COMPLETADA")
        print(f"{'='*60}")
        
        return results
    
    def _analyze_simulation(self, ST, scenario_name):
        """
        Analiza resultados de simulación.
        
        Args:
            ST: Array con precios finales
            scenario_name: Nombre del escenario
            
        Returns:
            dict con estadísticas
        """
        # Estadísticas básicas
        mean_price = np.mean(ST)
        median_price = np.median(ST)
        std_price = np.std(ST)
        
        # Percentiles
        p05 = np.percentile(ST, 5)
        p25 = np.percentile(ST, 25)
        p75 = np.percentile(ST, 75)
        p95 = np.percentile(ST, 95)
        
        # Retornos
        returns = (ST - self.S0) / self.S0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Probabilidad de pérdida
        prob_loss = np.mean(ST < self.S0)
        
        # Value at Risk (VaR) y Expected Shortfall (ES)
        var_95 = self.S0 - p05
        losses = self.S0 - ST[ST < self.S0]
        es_95 = np.mean(losses) if len(losses) > 0 else 0
        
        print(f"\n  📊 Estadísticas:")
        print(f"     • Media: ${mean_price:.2f} (retorno: {mean_return:.2%})")
        print(f"     • Mediana: ${median_price:.2f}")
        print(f"     • Std Dev: ${std_price:.2f}")
        print(f"     • P5-P95: ${p05:.2f} - ${p95:.2f}")
        print(f"     • Prob(pérdida): {prob_loss:.2%}")
        print(f"     • VaR(95%): ${var_95:.2f}")
        print(f"     • ES(95%): ${es_95:.2f}")
        
        return {
            'scenario': scenario_name,
            'prices': ST,
            'mean_price': mean_price,
            'median_price': median_price,
            'std_price': std_price,
            'p05': p05,
            'p25': p25,
            'p75': p75,
            'p95': p95,
            'mean_return': mean_return,
            'std_return': std_return,
            'prob_loss': prob_loss,
            'var_95': var_95,
            'es_95': es_95
        }
    
    def calculate_scenario_entropies(self):
        """
        Calcula entropías para cada escenario simulado.
        
        Returns:
            DataFrame con entropías por escenario
        """
        print(f"\n{'='*60}")
        print("CALCULANDO ENTROPÍAS DE ESCENARIOS")
        print(f"{'='*60}")
        
        from entropy_calculator import EntropyCalculator
        
        entropy_calc = EntropyCalculator()
        results = []
        
        for scenario_name, data in self.simulation_results.items():
            print(f"\n  {scenario_name}:")
            
            # Estimar densidad con KDE
            prices = data['prices']
            kde = gaussian_kde(prices, bw_method='scott')
            
            # Crear grilla
            p_min = np.percentile(prices, 0.1)
            p_max = np.percentile(prices, 99.9)
            price_grid = np.linspace(p_min, p_max, 500)
            
            # Densidad
            density = kde(price_grid)
            density = density / np.trapz(density, price_grid)  # Normalizar
            
            # Shannon entropy
            H_shannon = entropy_calc.shannon_entropy(density, price_grid)
            
            # Rényi entropies
            H_renyi_05 = entropy_calc.renyi_entropy(density, price_grid, 0.5)
            H_renyi_15 = entropy_calc.renyi_entropy(density, price_grid, 1.5)
            H_renyi_20 = entropy_calc.renyi_entropy(density, price_grid, 2.0)
            
            print(f"     • H_shannon: {H_shannon:.4f}")
            print(f"     • H_rényi(0.5): {H_renyi_05:.4f}")
            print(f"     • H_rényi(1.5): {H_renyi_15:.4f}")
            print(f"     • H_rényi(2.0): {H_renyi_20:.4f}")
            
            results.append({
                'scenario': scenario_name,
                'H_shannon': H_shannon,
                'H_renyi_05': H_renyi_05,
                'H_renyi_15': H_renyi_15,
                'H_renyi_20': H_renyi_20,
                'mean_price': data['mean_price'],
                'std_price': data['std_price']
            })
        
        df_entropy = pd.DataFrame(results)
        
        print(f"\n✅ Entropías calculadas para {len(results)} escenarios")
        
        return df_entropy
    
    def save_results(self, output_dir='output'):
        """
        Guarda resultados de simulaciones.
        
        Args:
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar resultados completos (pickle)
        pickle_file = output_path / 'monte_carlo_results.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.simulation_results, f)
        print(f"\n💾 Resultados guardados en: {pickle_file}")
        
        # Guardar resumen estadístico (CSV)
        summary_data = []
        for scenario_name, data in self.simulation_results.items():
            summary_data.append({
                'scenario': data['scenario'],
                'mean_price': data['mean_price'],
                'median_price': data['median_price'],
                'std_price': data['std_price'],
                'p05': data['p05'],
                'p95': data['p95'],
                'mean_return': data['mean_return'],
                'prob_loss': data['prob_loss'],
                'var_95': data['var_95'],
                'es_95': data['es_95']
            })
        
        df_summary = pd.DataFrame(summary_data)
        csv_file = output_path / 'monte_carlo_summary.csv'
        df_summary.to_csv(csv_file, index=False)
        print(f"💾 Resumen guardado en: {csv_file}")
    
    def plot_simulation_results(self, output_dir='output/plots'):
        """
        Grafica resultados de simulaciones.
        
        Args:
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GRAFICANDO RESULTADOS DE SIMULACIÓN")
        print(f"{'='*60}")
        
        # Figura 1: Histogramas de distribuciones
        self._plot_distributions(output_path)
        
        # Figura 2: Comparación de estadísticas
        self._plot_statistics_comparison(output_path)
        
        # Figura 3: Risk metrics
        self._plot_risk_metrics(output_path)
    
    def _plot_distributions(self, output_path):
        """Grafica distribuciones de precios simulados."""
        n_scenarios = len(self.simulation_results)
        n_cols = 3
        n_rows = (n_scenarios + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_scenarios > 1 else [axes]
        
        for idx, (scenario_name, data) in enumerate(self.simulation_results.items()):
            ax = axes[idx]
            
            prices = data['prices']
            
            # Histograma
            ax.hist(prices, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')
            
            # KDE
            kde = gaussian_kde(prices, bw_method='scott')
            x_grid = np.linspace(prices.min(), prices.max(), 200)
            ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE')
            
            # Líneas de referencia
            ax.axvline(self.S0, color='black', linestyle='--', linewidth=2, 
                      label=f'S0=${self.S0:.2f}')
            ax.axvline(data['mean_price'], color='green', linestyle=':', linewidth=2,
                      label=f'Mean=${data["mean_price"]:.2f}')
            
            ax.set_xlabel('Precio Final ($)', fontsize=10)
            ax.set_ylabel('Densidad', fontsize=10)
            ax.set_title(f"{data['scenario']}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes extras
        for idx in range(n_scenarios, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Distribuciones de Precios - Escenarios Monte Carlo', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_path / 'mc_distributions.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_file}")
        
        plt.close()
    
    def _plot_statistics_comparison(self, output_path):
        """Grafica comparación de estadísticas entre escenarios."""
        scenarios = []
        means = []
        stds = []
        p05s = []
        p95s = []
        
        for scenario_name, data in self.simulation_results.items():
            scenarios.append(data['scenario'])
            means.append(data['mean_price'])
            stds.append(data['std_price'])
            p05s.append(data['p05'])
            p95s.append(data['p95'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Media y desviación estándar
        ax1 = axes[0, 0]
        x_pos = np.arange(len(scenarios))
        ax1.bar(x_pos, means, alpha=0.7, color='blue', edgecolor='black')
        ax1.axhline(self.S0, color='red', linestyle='--', linewidth=2, label=f'S0=${self.S0:.2f}')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Precio Medio ($)', fontsize=11)
        ax1.set_title('Precio Medio por Escenario', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Desviación estándar
        ax2 = axes[0, 1]
        ax2.bar(x_pos, stds, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Desviación Estándar ($)', fontsize=11)
        ax2.set_title('Volatilidad por Escenario', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Rangos de confianza
        ax3 = axes[1, 0]
        for i in range(len(scenarios)):
            ax3.plot([i, i], [p05s[i], p95s[i]], 'o-', linewidth=3, markersize=8)
            ax3.plot(i, means[i], 's', markersize=10, color='red')
        
        ax3.axhline(self.S0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Precio ($)', fontsize=11)
        ax3.set_title('Intervalos de Confianza (P5-P95)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(['P5-P95', 'Media', 'S0'], loc='best')
        
        # Retornos esperados
        ax4 = axes[1, 1]
        returns = [(m - self.S0) / self.S0 * 100 for m in means]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax4.bar(x_pos, returns, alpha=0.7, color=colors, edgecolor='black')
        ax4.axhline(0, color='black', linewidth=2)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Retorno (%)', fontsize=11)
        ax4.set_title('Retorno Esperado por Escenario', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Comparación de Estadísticas - Escenarios Monte Carlo', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_path / 'mc_statistics.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {output_file}")
        
        plt.close()
    
    def _plot_risk_metrics(self, output_path):
        """Grafica métricas de riesgo."""
        scenarios = []
        prob_losses = []
        vars = []
        ess = []
        
        for scenario_name, data in self.simulation_results.items():
            scenarios.append(data['scenario'])
            prob_losses.append(data['prob_loss'] * 100)
            vars.append(data['var_95'])
            ess.append(data['es_95'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        x_pos = np.arange(len(scenarios))
        
        # Probabilidad de pérdida
        ax1 = axes[0]
        ax1.bar(x_pos, prob_losses, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Probabilidad (%)', fontsize=11)
        ax1.set_title('Probabilidad de Pérdida', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Value at Risk (95%)
        ax2 = axes[1]
        ax2.bar(x_pos, vars, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('VaR ($)', fontsize=11)
        ax2.set_title('Value at Risk (95%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Expected Shortfall (95%)
        ax3 = axes[2]
        ax3.bar(x_pos, ess, alpha=0.7, color='darkred', edgecolor='black')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('ES ($)', fontsize=11)
        ax3.set_title('Expected Shortfall (95%)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Métricas de Riesgo - Escenarios Monte Carlo', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_file = output_path / 'mc_risk_metrics.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {output_file}")
        
        plt.close()


def create_stress_scenarios(historical_sigma):
    """
    Crea configuraciones de escenarios de estrés.
    
    Args:
        historical_sigma: Volatilidad histórica anualizada
        
    Returns:
        Lista de configuraciones de escenarios
    """
    scenarios = [
        {
            'name': 'Alta Volatilidad (BS)',
            'model': 'black_scholes',
            'sigma': historical_sigma * 1.5
        },
        {
            'name': 'Volatilidad Extrema (BS)',
            'model': 'black_scholes',
            'sigma': historical_sigma * 2.0
        },
        {
            'name': 'Vol Estocástica (Heston)',
            'model': 'heston',
            'v0': historical_sigma**2,
            'kappa': 2.0,
            'theta': historical_sigma**2 * 1.2,
            'sigma_v': 0.3,
            'rho': -0.7
        },
        {
            'name': 'Saltos Negativos (Merton)',
            'model': 'merton_jump',
            'sigma': historical_sigma * 0.8,
            'lambda_jump': 5.0,
            'mu_jump': -0.05,
            'sigma_jump': 0.03
        },
        {
            'name': 'Crisis Severa (Merton)',
            'model': 'merton_jump',
            'sigma': historical_sigma * 1.2,
            'lambda_jump': 10.0,
            'mu_jump': -0.10,
            'sigma_jump': 0.05
        }
    ]
    
    return scenarios


def main():
    """
    Función principal para simulación Monte Carlo.
    """
    from pathlib import Path
    import pickle
    
    # Cargar datos históricos para obtener volatilidad
    hist_file = Path('data') / 'AAPL_historical.csv'
    options_file = Path('data') / 'AAPL_options_clean.csv'
    
    if not hist_file.exists() or not options_file.exists():
        print(f"❌ Error: Faltan archivos de datos")
        print("   Ejecuta primero data_loader.py y preprocessing.py")
        return
    
    print(f"📂 Cargando datos...")
    
    # Obtener precio actual
    df_options = pd.read_csv(options_file)
    S0 = df_options['stockPrice'].iloc[0]
    
    # Obtener volatilidad histórica
    df_hist = pd.read_csv(hist_file, index_col=0, parse_dates=True)
    if 'log_return' not in df_hist.columns:
        df_hist['log_return'] = np.log(df_hist['Close'] / df_hist['Close'].shift(1))
    
    hist_sigma = df_hist['log_return'].std() * np.sqrt(252)
    
    print(f"\n📊 Parámetros:")
    print(f"  • S0: ${S0:.2f}")
    print(f"  • Volatilidad histórica anualizada: {hist_sigma:.2%}")
    
    # Crear simulador
    simulator = MonteCarloSimulator(
        S0=S0,
        r=0.045,
        n_simulations=10000,
        seed=42
    )
    
    # Configurar escenarios
    base_params = {'sigma': hist_sigma}
    stress_scenarios = create_stress_scenarios(hist_sigma)
    
    # Ejecutar simulaciones (horizonte de 30 días)
    T = 30 / 365.0
    results = simulator.run_stress_scenarios(T, base_params, stress_scenarios)
    
    # Calcular entropías de escenarios
    df_scenario_entropy = simulator.calculate_scenario_entropies()
    
    # Guardar resultados
    simulator.save_results(output_dir='output')
    
    # Guardar entropías de escenarios
    entropy_file = Path('output') / 'scenario_entropies.csv'
    df_scenario_entropy.to_csv(entropy_file, index=False)
    print(f"\n💾 Entropías de escenarios guardadas en: {entropy_file}")
    
    # Graficar
    simulator.plot_simulation_results(output_dir='output/plots')
    
    print("\n✅ Simulación Monte Carlo completada")


if __name__ == "__main__":
    main()