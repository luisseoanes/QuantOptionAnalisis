"""
density_reconstruction.py
Módulo para reconstruir la densidad riesgo-neutral (RND) del mercado.
Implementa métodos Breeden-Litzenberger y MaxEnt.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RiskNeutralDensity:
    """
    Reconstruye la densidad riesgo-neutral a partir de precios de opciones.
    """
    
    def __init__(self, risk_free_rate=0.045, smoothing=0.01, n_grid=500):
        """
        Inicializa el reconstructor de densidad.
        
        Args:
            risk_free_rate: Tasa libre de riesgo
            smoothing: Parámetro de suavizado para splines
            n_grid: Número de puntos en la grilla
        """
        self.r = risk_free_rate
        self.smoothing = smoothing
        self.n_grid = n_grid
        self.densities = {}  # Almacena densidades por vencimiento
    
    def breeden_litzenberger(self, df_options, expiration):
        """
        Método Breeden-Litzenberger (model-free) para estimar RND.
        
        Usa la segunda derivada de precios de call respecto al strike:
        q(K) = exp(rT) * d²C/dK²
        
        Args:
            df_options: DataFrame con opciones
            expiration: Fecha de vencimiento específica
            
        Returns:
            dict con 'strikes', 'density', 'cum_density', 'T'
        """
        print(f"\n{'='*60}")
        print(f"BREEDEN-LITZENBERGER: {expiration.date()}")
        print(f"{'='*60}")
        
        # Filtrar calls para este vencimiento
        mask = (df_options['expiration'] == expiration) & (df_options['type'] == 'call')
        df_calls = df_options[mask].copy()
        
        if len(df_calls) < 5:
            print(f"❌ Insuficientes calls ({len(df_calls)}) para reconstrucción")
            return None
        
        # Ordenar por strike
        df_calls = df_calls.sort_values('strike')
        
        # Obtener datos
        strikes = df_calls['strike'].values
        prices = df_calls['mid_price'].values
        T = df_calls['T_years'].iloc[0]
        S = df_calls['stockPrice'].iloc[0]
        
        print(f"\n📊 Datos:")
        print(f"  • Calls disponibles: {len(df_calls)}")
        print(f"  • Rango de strikes: ${strikes.min():.2f} - ${strikes.max():.2f}")
        print(f"  • Tiempo a vencimiento: {T:.4f} años ({T*365:.0f} días)")
        print(f"  • Precio actual: ${S:.2f}")
        
        # Interpolar precios con spline
        print(f"\n🔄 Interpolando precios con spline (s={self.smoothing})...")
        
        try:
            # Usar UnivariateSpline con suavizado
            spline = UnivariateSpline(strikes, prices, s=self.smoothing, k=3)
            
            # Crear grilla fina de strikes
            k_min = max(strikes.min(), S * 0.5)
            k_max = min(strikes.max(), S * 1.5)
            strike_grid = np.linspace(k_min, k_max, self.n_grid)
            
            # Evaluar spline
            call_prices_smooth = spline(strike_grid)
            
            # Asegurar monotonicidad (calls deben decrecer con K)
            for i in range(1, len(call_prices_smooth)):
                if call_prices_smooth[i] > call_prices_smooth[i-1]:
                    call_prices_smooth[i] = call_prices_smooth[i-1]
            
            # Calcular segunda derivada numérica
            print("🔄 Calculando segunda derivada...")
            
            # Método de diferencias finitas central
            dk = strike_grid[1] - strike_grid[0]
            d2C_dK2 = np.zeros_like(strike_grid)
            
            for i in range(1, len(strike_grid) - 1):
                d2C_dK2[i] = (call_prices_smooth[i+1] - 2*call_prices_smooth[i] + 
                             call_prices_smooth[i-1]) / (dk**2)
            
            # Endpoints usando diferencias hacia adelante/atrás
            d2C_dK2[0] = d2C_dK2[1]
            d2C_dK2[-1] = d2C_dK2[-2]
            
            # Densidad riesgo-neutral
            density = np.exp(self.r * T) * d2C_dK2
            
            # Limpiar: eliminar negativos
            density = np.maximum(density, 0)
            
            # Normalizar para que integre a 1
            area = np.trapz(density, strike_grid)
            
            if area > 0:
                density = density / area
                print(f"  ✓ Densidad normalizada (área original: {area:.6f})")
            else:
                print(f"  ⚠️  Área de densidad es cero o negativa")
                return None
            
            # Calcular CDF
            cum_density = np.cumsum(density) * (strike_grid[1] - strike_grid[0])
            cum_density = cum_density / cum_density[-1]  # Normalizar a [0,1]
            
            # Calcular momentos
            mean_rn = np.trapz(strike_grid * density, strike_grid)
            var_rn = np.trapz((strike_grid - mean_rn)**2 * density, strike_grid)
            std_rn = np.sqrt(var_rn)
            
            print(f"\n📊 Estadísticas de RND:")
            print(f"  • Media: ${mean_rn:.2f}")
            print(f"  • Std Dev: ${std_rn:.2f}")
            print(f"  • Coef. Variación: {std_rn/mean_rn:.2%}")
            
            # Percentiles
            p10 = strike_grid[np.searchsorted(cum_density, 0.10)]
            p50 = strike_grid[np.searchsorted(cum_density, 0.50)]
            p90 = strike_grid[np.searchsorted(cum_density, 0.90)]
            
            print(f"  • P10: ${p10:.2f}")
            print(f"  • P50: ${p50:.2f}")
            print(f"  • P90: ${p90:.2f}")
            
            result = {
                'strikes': strike_grid,
                'density': density,
                'cum_density': cum_density,
                'call_prices': call_prices_smooth,
                'T': T,
                'S': S,
                'mean': mean_rn,
                'std': std_rn,
                'method': 'breeden_litzenberger'
            }
            
            print(f"\n✅ Reconstrucción exitosa")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en reconstrucción: {str(e)}")
            return None
    
    def maxent_density(self, df_options, expiration, n_moments=5):
        """
        Método de Máxima Entropía (MaxEnt) para estimar RND.
        
        Maximiza entropía sujeto a restricciones de precios de opciones.
        
        Args:
            df_options: DataFrame con opciones
            expiration: Fecha de vencimiento
            n_moments: Número de momentos/restricciones
            
        Returns:
            dict con densidad MaxEnt
        """
        print(f"\n{'='*60}")
        print(f"MAXENT DENSITY: {expiration.date()}")
        print(f"{'='*60}")
        
        # Filtrar opciones para este vencimiento
        mask = df_options['expiration'] == expiration
        df_exp = df_options[mask].copy()
        
        if len(df_exp) < n_moments:
            print(f"❌ Insuficientes opciones ({len(df_exp)}) para MaxEnt")
            return None
        
        T = df_exp['T_years'].iloc[0]
        S = df_exp['stockPrice'].iloc[0]
        
        # Seleccionar opciones representativas (calls ATM y OTM)
        df_calls = df_exp[df_exp['type'] == 'call'].sort_values('strike')
        
        if len(df_calls) < n_moments:
            n_moments = len(df_calls)
        
        # Seleccionar strikes espaciados
        indices = np.linspace(0, len(df_calls)-1, n_moments, dtype=int)
        selected_calls = df_calls.iloc[indices]
        
        strikes_obs = selected_calls['strike'].values
        prices_obs = selected_calls['mid_price'].values
        
        print(f"\n📊 Configuración:")
        print(f"  • Restricciones (momentos): {n_moments}")
        print(f"  • Strikes seleccionados: {strikes_obs}")
        
        # Definir grilla de strikes
        k_min = S * 0.5
        k_max = S * 1.5
        strike_grid = np.linspace(k_min, k_max, self.n_grid)
        dk = strike_grid[1] - strike_grid[0]
        
        # Función objetivo: minimizar entropía negativa (maximizar entropía)
        def objective(lambdas):
            """Lagrangiano: -H + sum(lambda_i * constraint_i)"""
            # Calcular densidad con multiplicadores de Lagrange
            log_q = -lambdas[0]  # Normalización
            
            for i, K in enumerate(strikes_obs):
                # Payoff de call: max(S_T - K, 0)
                payoff = np.maximum(strike_grid - K, 0)
                log_q -= lambdas[i+1] * payoff
            
            # Densidad
            q = np.exp(log_q)
            q = q / (np.sum(q) * dk)  # Normalizar
            
            # Entropía (negativa para minimizar)
            entropy = -np.sum(q * np.log(q + 1e-10) * dk)
            
            # Restricciones: diferencias entre precios teóricos y observados
            penalty = 0
            for i, K in enumerate(strikes_obs):
                payoff = np.maximum(strike_grid - K, 0)
                theo_price = np.exp(-self.r * T) * np.sum(q * payoff * dk)
                penalty += (theo_price - prices_obs[i])**2
            
            return entropy + 1000 * penalty  # Penalización fuerte
        
        print("\n🔄 Optimizando multiplicadores de Lagrange...")
        
        # Optimizar
        lambda0 = np.zeros(n_moments + 1)
        
        try:
            result = minimize(objective, lambda0, method='BFGS',
                            options={'maxiter': 500, 'disp': False})
            
            if not result.success:
                print(f"  ⚠️  Optimización no convergió: {result.message}")
            
            lambdas_opt = result.x
            
            # Calcular densidad óptima
            log_q = -lambdas_opt[0]
            for i, K in enumerate(strikes_obs):
                payoff = np.maximum(strike_grid - K, 0)
                log_q -= lambdas_opt[i+1] * payoff
            
            density = np.exp(log_q)
            density = density / (np.sum(density) * dk)
            
            # Validar
            if np.any(np.isnan(density)) or np.any(density < 0):
                print("❌ Densidad inválida")
                return None
            
            # Estadísticas
            mean_rn = np.sum(strike_grid * density * dk)
            var_rn = np.sum((strike_grid - mean_rn)**2 * density * dk)
            std_rn = np.sqrt(var_rn)
            
            print(f"\n📊 Estadísticas de MaxEnt RND:")
            print(f"  • Media: ${mean_rn:.2f}")
            print(f"  • Std Dev: ${std_rn:.2f}")
            
            result_dict = {
                'strikes': strike_grid,
                'density': density,
                'T': T,
                'S': S,
                'mean': mean_rn,
                'std': std_rn,
                'lambdas': lambdas_opt,
                'method': 'maxent'
            }
            
            print(f"✅ MaxEnt completado")
            
            return result_dict
            
        except Exception as e:
            print(f"❌ Error en MaxEnt: {str(e)}")
            return None
    
    def reconstruct_all_maturities(self, df_options):
        """
        Reconstruye densidades para todos los vencimientos.
        
        Args:
            df_options: DataFrame con opciones
            
        Returns:
            dict con densidades por vencimiento
        """
        print(f"\n{'='*70}")
        print("RECONSTRUCCIÓN DE DENSIDADES - TODOS LOS VENCIMIENTOS")
        print(f"{'='*70}")
        
        expirations = sorted(df_options['expiration'].unique())
        print(f"\n📅 Vencimientos a procesar: {len(expirations)}")
        
        densities = {}
        
        for exp in expirations:
            print(f"\n{'─'*70}")
            print(f"Procesando: {exp.date()}")
            
            # Método Breeden-Litzenberger
            rnd_bl = self.breeden_litzenberger(df_options, exp)
            
            # Método MaxEnt
            rnd_maxent = self.maxent_density(df_options, exp, n_moments=5)
            
            if rnd_bl is not None or rnd_maxent is not None:
                densities[exp] = {
                    'breeden_litzenberger': rnd_bl,
                    'maxent': rnd_maxent
                }
        
        self.densities = densities
        
        print(f"\n{'='*70}")
        print(f"✅ RECONSTRUCCIÓN COMPLETADA")
        print(f"   Densidades exitosas: {len(densities)}/{len(expirations)}")
        print(f"{'='*70}")
        
        return densities
    
    def save_densities(self, output_dir='output'):
        """
        Guarda densidades en archivos.
        
        Args:
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar como pickle
        pickle_file = output_path / 'rnd_densities.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.densities, f)
        
        print(f"\n💾 Densidades guardadas en: {pickle_file}")
        
        # Guardar resumen en CSV
        summary_data = []
        
        for exp, methods in self.densities.items():
            for method_name, rnd in methods.items():
                if rnd is not None:
                    summary_data.append({
                        'expiration': exp,
                        'T_days': rnd['T'] * 365,
                        'T_years': rnd['T'],
                        'method': method_name,
                        'mean': rnd['mean'],
                        'std': rnd['std'],
                        'S_current': rnd['S']
                    })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            csv_file = output_path / 'rnd_summary.csv'
            df_summary.to_csv(csv_file, index=False)
            print(f"💾 Resumen guardado en: {csv_file}")
    
    def plot_densities(self, output_dir='output/plots'):
        """
        Grafica densidades reconstruidas.
        
        Args:
            output_dir: Directorio para gráficos
        """
        if not self.densities:
            print("❌ No hay densidades para graficar")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GRAFICANDO DENSIDADES")
        print(f"{'='*60}")
        
        # Seleccionar los 4 vencimientos más cercanos con datos válidos
        expirations = sorted(self.densities.keys())[:4]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, exp in enumerate(expirations):
            ax = axes[idx]
            methods = self.densities[exp]
            
            # Obtener datos
            rnd_bl = methods.get('breeden_litzenberger')
            rnd_maxent = methods.get('maxent')
            
            if rnd_bl is None and rnd_maxent is None:
                continue
            
            # Graficar Breeden-Litzenberger
            if rnd_bl is not None:
                ax.plot(rnd_bl['strikes'], rnd_bl['density'], 
                       'b-', linewidth=2, label='Breeden-Litzenberger', alpha=0.7)
                ax.axvline(rnd_bl['S'], color='black', linestyle='--', 
                          alpha=0.5, linewidth=1.5, label=f"S = ${rnd_bl['S']:.2f}")
                T_days = int(rnd_bl['T'] * 365)
            
            # Graficar MaxEnt
            if rnd_maxent is not None:
                ax.plot(rnd_maxent['strikes'], rnd_maxent['density'], 
                       'r--', linewidth=2, label='MaxEnt', alpha=0.7)
                if rnd_bl is None:
                    ax.axvline(rnd_maxent['S'], color='black', linestyle='--', 
                              alpha=0.5, linewidth=1.5, label=f"S = ${rnd_maxent['S']:.2f}")
                    T_days = int(rnd_maxent['T'] * 365)
            
            ax.set_xlabel('Strike Price ($)', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'RND - Exp: {exp.date()} ({T_days} días)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Risk-Neutral Densities (RND) - AAPL', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Guardar
        output_file = output_path / 'rnd_densities.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_file}")
        
        plt.close()
        
        # Gráfico comparativo: CDF
        self._plot_cumulative_distributions(output_path)
    
    def _plot_cumulative_distributions(self, output_path):
        """
        Grafica distribuciones acumuladas (CDF).
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        expirations = sorted(self.densities.keys())[:4]
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, exp in enumerate(expirations):
            rnd_bl = self.densities[exp].get('breeden_litzenberger')
            
            if rnd_bl is not None and 'cum_density' in rnd_bl:
                T_days = int(rnd_bl['T'] * 365)
                ax.plot(rnd_bl['strikes'], rnd_bl['cum_density'], 
                       color=colors[idx], linewidth=2, 
                       label=f"{exp.date()} ({T_days}d)", alpha=0.7)
        
        ax.set_xlabel('Strike Price ($)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Risk-Neutral Cumulative Distribution Functions', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        output_file = output_path / 'rnd_cdf.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico CDF guardado: {output_file}")
        
        plt.close()


class PhysicalDensityEstimator:
    """
    Estima la densidad física (histórica) del activo.
    """
    
    def __init__(self, bandwidth='scott'):
        """
        Inicializa el estimador.
        
        Args:
            bandwidth: Método de bandwidth para KDE ('scott', 'silverman', o float)
        """
        self.bandwidth = bandwidth
        self.kde = None
        self.density_data = None
    
    def estimate_from_returns(self, df_historical, current_price, horizon_days=30):
        """
        Estima densidad física usando retornos históricos.
        
        Args:
            df_historical: DataFrame con precios históricos
            current_price: Precio actual del activo
            horizon_days: Horizonte temporal (días)
            
        Returns:
            dict con densidad física
        """
        print(f"\n{'='*60}")
        print("ESTIMACIÓN DE DENSIDAD FÍSICA")
        print(f"{'='*60}")
        
        # Calcular retornos logarítmicos
        if 'log_return' not in df_historical.columns:
            df_historical = df_historical.copy()
            df_historical['log_return'] = np.log(
                df_historical['Close'] / df_historical['Close'].shift(1)
            )
        
        # Filtrar retornos válidos
        returns = df_historical['log_return'].dropna().values
        
        print(f"\n📊 Datos históricos:")
        print(f"  • Observaciones: {len(returns)}")
        print(f"  • Precio actual: ${current_price:.2f}")
        print(f"  • Horizonte: {horizon_days} días")
        
        # Estadísticas de retornos
        mu_daily = returns.mean()
        sigma_daily = returns.std()
        
        print(f"\n📈 Estadísticas de retornos diarios:")
        print(f"  • Media: {mu_daily:.6f} ({mu_daily*252:.2%} anual)")
        print(f"  • Std Dev: {sigma_daily:.6f} ({sigma_daily*np.sqrt(252):.2%} anual)")
        
        # Escalar al horizonte
        T = horizon_days / 365.0
        mu_T = mu_daily * horizon_days
        sigma_T = sigma_daily * np.sqrt(horizon_days)
        
        print(f"\n📊 Parámetros escalados a {horizon_days} días:")
        print(f"  • Media: {mu_T:.6f}")
        print(f"  • Std Dev: {sigma_T:.6f}")
        
        # Simular precios futuros usando log-normal
        # S_T = S_0 * exp((mu - 0.5*sigma²)*T + sigma*sqrt(T)*Z)
        # Donde Z ~ N(0,1)
        
        # Crear grilla de precios futuros
        n_grid = 500
        S_min = current_price * 0.5
        S_max = current_price * 1.5
        price_grid = np.linspace(S_min, S_max, n_grid)
        
        # Calcular densidad usando KDE sobre retornos históricos
        print("\n🔄 Estimando densidad con KDE...")
        
        try:
            kde = gaussian_kde(returns, bw_method=self.bandwidth)
            self.kde = kde
            
            # Para cada precio en la grilla, calcular la densidad
            # p(S_T) = p(r) / S_T, donde r = log(S_T / S_0)
            log_returns_grid = np.log(price_grid / current_price)
            density_returns = kde(log_returns_grid)
            
            # Transformar a densidad de precios
            density_prices = density_returns / price_grid
            
            # Normalizar
            area = np.trapz(density_prices, price_grid)
            density_prices = density_prices / area
            
            # Calcular CDF
            cum_density = np.cumsum(density_prices) * (price_grid[1] - price_grid[0])
            cum_density = cum_density / cum_density[-1]
            
            # Estadísticas
            mean_phys = np.trapz(price_grid * density_prices, price_grid)
            var_phys = np.trapz((price_grid - mean_phys)**2 * density_prices, price_grid)
            std_phys = np.sqrt(var_phys)
            
            print(f"\n📊 Estadísticas de densidad física:")
            print(f"  • Media: ${mean_phys:.2f}")
            print(f"  • Std Dev: ${std_phys:.2f}")
            print(f"  • Coef. Variación: {std_phys/mean_phys:.2%}")
            
            self.density_data = {
                'strikes': price_grid,
                'density': density_prices,
                'cum_density': cum_density,
                'mean': mean_phys,
                'std': std_phys,
                'mu_daily': mu_daily,
                'sigma_daily': sigma_daily,
                'T': T,
                'S_current': current_price,
                'horizon_days': horizon_days
            }
            
            print(f"\n✅ Estimación física completada")
            
            return self.density_data
            
        except Exception as e:
            print(f"❌ Error en estimación: {str(e)}")
            return None
    
    def save_physical_density(self, output_dir='output'):
        """
        Guarda densidad física.
        
        Args:
            output_dir: Directorio de salida
        """
        if self.density_data is None:
            print("❌ No hay densidad física para guardar")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar pickle
        pickle_file = output_path / 'density_physical.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.density_data, f)
        
        print(f"\n💾 Densidad física guardada en: {pickle_file}")
    
    def plot_physical_density(self, output_dir='output/plots'):
        """
        Grafica densidad física.
        
        Args:
            output_dir: Directorio de salida
        """
        if self.density_data is None:
            print("❌ No hay densidad física para graficar")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PDF
        ax1.plot(self.density_data['strikes'], self.density_data['density'], 
                'g-', linewidth=2, label='Physical Density')
        ax1.axvline(self.density_data['S_current'], color='black', 
                   linestyle='--', linewidth=1.5, 
                   label=f"Current: ${self.density_data['S_current']:.2f}")
        ax1.axvline(self.density_data['mean'], color='red', 
                   linestyle=':', linewidth=1.5, 
                   label=f"Mean: ${self.density_data['mean']:.2f}")
        ax1.set_xlabel('Price ($)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f"Physical Density ({self.density_data['horizon_days']} days)", 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        ax2.plot(self.density_data['strikes'], self.density_data['cum_density'], 
                'g-', linewidth=2)
        ax2.axvline(self.density_data['S_current'], color='black', 
                   linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Price ($)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Physical CDF', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        output_file = output_path / 'density_physical.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_file}")
        
        plt.close()


def main():
    """
    Función principal para reconstrucción de densidades.
    """
    from pathlib import Path
    
    # Cargar datos de opciones
    options_file = Path('data') / 'AAPL_options_with_iv.csv'
    historical_file = Path('data') / 'AAPL_historical.csv'
    
    if not options_file.exists():
        print(f"❌ Error: No se encuentra {options_file}")
        print("   Ejecuta primero implied_vol.py")
        return
    
    print(f"📂 Cargando datos de opciones desde {options_file}")
    df_options = pd.read_csv(options_file)
    df_options['expiration'] = pd.to_datetime(df_options['expiration'])
    
    # Reconstruir densidades riesgo-neutral
    rnd = RiskNeutralDensity(risk_free_rate=0.045, smoothing=0.01, n_grid=500)
    densities = rnd.reconstruct_all_maturities(df_options)
    
    # Guardar y graficar
    rnd.save_densities(output_dir='output')
    rnd.plot_densities(output_dir='output/plots')
    
    # Estimar densidad física
    if historical_file.exists():
        print(f"\n📂 Cargando datos históricos desde {historical_file}")
        df_historical = pd.read_csv(historical_file, index_col=0, parse_dates=True)
        
        current_price = df_options['stockPrice'].iloc[0]
        
        phys_estimator = PhysicalDensityEstimator(bandwidth='scott')
        phys_density = phys_estimator.estimate_from_returns(
            df_historical, current_price, horizon_days=30
        )
        
        if phys_density:
            phys_estimator.save_physical_density(output_dir='output')
            phys_estimator.plot_physical_density(output_dir='output/plots')
    
    print("\n✅ Reconstrucción de densidades completada")


if __name__ == "__main__":
    main()