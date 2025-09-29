"""
implied_vol.py
Módulo para calcular volatilidad implícita usando Black-Scholes.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, newton
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BlackScholesModel:
    """
    Implementación del modelo Black-Scholes para opciones europeas.
    """
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calcula d1 de Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calcula d2 de Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """
        Calcula precio de una opción call europea.
        
        Args:
            S: Precio del activo subyacente
            K: Strike
            T: Tiempo hasta vencimiento (años)
            r: Tasa libre de riesgo
            sigma: Volatilidad
            
        Returns:
            Precio de la call
        """
        if T <= 0:
            return max(S - K, 0)
        
        if sigma <= 0:
            return max(S - K * np.exp(-r * T), 0)
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """
        Calcula precio de una opción put europea.
        
        Args:
            S: Precio del activo subyacente
            K: Strike
            T: Tiempo hasta vencimiento (años)
            r: Tasa libre de riesgo
            sigma: Volatilidad
            
        Returns:
            Precio de la put
        """
        if T <= 0:
            return max(K - S, 0)
        
        if sigma <= 0:
            return max(K * np.exp(-r * T) - S, 0)
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """
        Calcula vega (sensibilidad al cambio en volatilidad).
        
        Args:
            S: Precio del activo
            K: Strike
            T: Tiempo hasta vencimiento
            r: Tasa libre de riesgo
            sigma: Volatilidad
            
        Returns:
            Vega
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega


class ImpliedVolatilityCalculator:
    """
    Calcula volatilidad implícita a partir de precios de opciones.
    """
    
    def __init__(self, risk_free_rate=0.045):
        """
        Inicializa el calculador.
        
        Args:
            risk_free_rate: Tasa libre de riesgo anual
        """
        self.r = risk_free_rate
        self.bs_model = BlackScholesModel()
    
    def implied_volatility(self, option_price, S, K, T, option_type='call', 
                          method='brentq', tol=1e-6, max_iter=100):
        """
        Calcula volatilidad implícita usando root-finding.
        
        Args:
            option_price: Precio observado de la opción
            S: Precio del subyacente
            K: Strike
            T: Tiempo hasta vencimiento (años)
            option_type: 'call' o 'put'
            method: 'brentq' o 'newton'
            tol: Tolerancia
            max_iter: Máximo de iteraciones
            
        Returns:
            Volatilidad implícita (o np.nan si falla)
        """
        # Validaciones
        if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
            return np.nan
        
        # Verificar arbitraje obvio
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        if option_price < intrinsic:
            return np.nan
        
        # Función objetivo
        def objective(sigma):
            if option_type == 'call':
                theo_price = self.bs_model.call_price(S, K, T, self.r, sigma)
            else:
                theo_price = self.bs_model.put_price(S, K, T, self.r, sigma)
            return theo_price - option_price
        
        try:
            if method == 'brentq':
                # Método Brent (robusto, requiere bracketing)
                iv = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
            else:
                # Método Newton (rápido, usa vega)
                def objective_newton(sigma):
                    return objective(sigma)
                
                def fprime(sigma):
                    return self.bs_model.vega(S, K, T, self.r, sigma)
                
                # Estimación inicial: fórmula aproximada de Brenner-Subrahmanyam
                iv_guess = np.sqrt(2 * np.pi / T) * option_price / S
                iv_guess = np.clip(iv_guess, 0.05, 3.0)
                
                iv = newton(objective_newton, x0=iv_guess, fprime=fprime, 
                           tol=tol, maxiter=max_iter)
            
            # Validar resultado
            if iv < 0.001 or iv > 5.0:
                return np.nan
            
            return iv
            
        except Exception:
            return np.nan
    
    def calculate_implied_volatilities(self, df_options):
        """
        Calcula IV para todas las opciones en el DataFrame.
        
        Args:
            df_options: DataFrame con datos de opciones
            
        Returns:
            DataFrame con columna 'impliedVol_calculated'
        """
        print(f"\n{'='*60}")
        print("CALCULANDO VOLATILIDADES IMPLÍCITAS")
        print(f"{'='*60}")
        
        df = df_options.copy()
        
        # Si ya existe impliedVolatility, usarla como referencia
        has_existing_iv = 'impliedVolatility' in df.columns
        
        if has_existing_iv:
            print("\n✓ Columna 'impliedVolatility' existente detectada")
            # Verificar cuántas son válidas
            n_valid = df['impliedVolatility'].notna().sum()
            print(f"  • IVs válidas existentes: {n_valid}/{len(df)}")
        
        # Calcular IV para todas las opciones
        print("\n🔄 Calculando IVs con Black-Scholes...")
        
        iv_list = []
        failed = 0
        
        for idx, row in df.iterrows():
            S = row['stockPrice']
            K = row['strike']
            T = row['T_years']
            price = row['mid_price']
            opt_type = row['type']
            
            iv = self.implied_volatility(price, S, K, T, opt_type)
            iv_list.append(iv)
            
            if np.isnan(iv):
                failed += 1
        
        df['impliedVol_calculated'] = iv_list
        
        print(f"\n✅ Cálculo completado:")
        print(f"  • IVs calculadas exitosamente: {len(df) - failed}/{len(df)}")
        print(f"  • Fallos: {failed}")
        
        # Usar IV calculada o existente
        if has_existing_iv:
            # Priorizar existente, llenar con calculada donde falte
            df['impliedVol'] = df['impliedVolatility'].fillna(df['impliedVol_calculated'])
            n_filled = df['impliedVolatility'].isna().sum()
            print(f"  • Rellenadas {n_filled} IVs faltantes con cálculo")
        else:
            df['impliedVol'] = df['impliedVol_calculated']
        
        # Estadísticas
        iv_valid = df['impliedVol'].dropna()
        if len(iv_valid) > 0:
            print(f"\n📊 Estadísticas de IV:")
            print(f"  • Media: {iv_valid.mean():.2%}")
            print(f"  • Mediana: {iv_valid.median():.2%}")
            print(f"  • Std: {iv_valid.std():.2%}")
            print(f"  • Min: {iv_valid.min():.2%}")
            print(f"  • Max: {iv_valid.max():.2%}")
        
        return df
    
    def plot_volatility_smile(self, df_options, output_dir='output/plots'):
        """
        Grafica el volatility smile por vencimiento.
        
        Args:
            df_options: DataFrame con IVs calculadas
            output_dir: Directorio para guardar gráficos
        """
        print(f"\n{'='*60}")
        print("GRAFICANDO VOLATILITY SMILE")
        print(f"{'='*60}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filtrar opciones con IV válida
        df = df_options[df_options['impliedVol'].notna()].copy()
        
        if len(df) == 0:
            print("❌ No hay datos válidos para graficar")
            return
        
        # Seleccionar los vencimientos más líquidos (top 4)
        exp_counts = df.groupby('expiration').size().sort_values(ascending=False)
        top_expirations = exp_counts.head(4).index
        
        df_plot = df[df['expiration'].isin(top_expirations)].copy()
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, exp_date in enumerate(top_expirations):
            ax = axes[idx]
            
            df_exp = df_plot[df_plot['expiration'] == exp_date]
            T_days = df_exp['T_days'].iloc[0]
            
            # Separar calls y puts
            calls = df_exp[df_exp['type'] == 'call']
            puts = df_exp[df_exp['type'] == 'put']
            
            # Graficar
            if len(calls) > 0:
                ax.scatter(calls['moneyness'], calls['impliedVol'], 
                          alpha=0.6, s=50, label='Calls', color='green')
            
            if len(puts) > 0:
                ax.scatter(puts['moneyness'], puts['impliedVol'], 
                          alpha=0.6, s=50, label='Puts', color='red')
            
            # Línea vertical en ATM
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='ATM')
            
            ax.set_xlabel('Moneyness (K/S)', fontsize=11)
            ax.set_ylabel('Implied Volatility', fontsize=11)
            ax.set_title(f'Expiration: {exp_date.date()} ({T_days} días)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.7, 1.3)
        
        plt.suptitle('Volatility Smile - AAPL Options', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Guardar
        output_file = output_path / 'volatility_smile.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_file}")
        
        plt.close()
        
        # Gráfico adicional: IV vs Strike para el vencimiento más cercano
        self._plot_iv_by_strike(df, output_path)
    
    def _plot_iv_by_strike(self, df, output_path):
        """
        Gráfico de IV vs Strike para el vencimiento más cercano.
        """
        # Vencimiento más cercano
        nearest_exp = df['expiration'].min()
        df_nearest = df[df['expiration'] == nearest_exp].copy()
        
        if len(df_nearest) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calls
        calls = df_nearest[df_nearest['type'] == 'call'].sort_values('strike')
        if len(calls) > 0:
            ax.plot(calls['strike'], calls['impliedVol'], 
                   'o-', label='Calls', color='green', markersize=6, alpha=0.7)
        
        # Puts
        puts = df_nearest[df_nearest['type'] == 'put'].sort_values('strike')
        if len(puts) > 0:
            ax.plot(puts['strike'], puts['impliedVol'], 
                   'o-', label='Puts', color='red', markersize=6, alpha=0.7)
        
        # Precio actual
        S = df_nearest['stockPrice'].iloc[0]
        ax.axvline(x=S, color='black', linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'Stock Price: ${S:.2f}')
        
        ax.set_xlabel('Strike Price ($)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        ax.set_title(f'Implied Volatility by Strike - Exp: {nearest_exp.date()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_path / 'iv_by_strike.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {output_file}")
        
        plt.close()


def main():
    """
    Función principal para calcular IVs.
    """
    from pathlib import Path
    
    # Cargar datos limpios
    data_file = Path('data') / 'AAPL_options_clean.csv'
    
    if not data_file.exists():
        print(f"❌ Error: No se encuentra {data_file}")
        print("   Ejecuta primero preprocessing.py")
        return
    
    print(f"📂 Cargando datos desde {data_file}")
    df_options = pd.read_csv(data_file)
    df_options['expiration'] = pd.to_datetime(df_options['expiration'])
    
    # Crear calculador
    iv_calculator = ImpliedVolatilityCalculator(risk_free_rate=0.045)
    
    # Calcular IVs
    df_with_iv = iv_calculator.calculate_implied_volatilities(df_options)
    
    # Guardar
    output_file = Path('data') / 'AAPL_options_with_iv.csv'
    df_with_iv.to_csv(output_file, index=False)
    print(f"\n💾 Datos con IV guardados en: {output_file}")
    
    # Graficar
    iv_calculator.plot_volatility_smile(df_with_iv)
    
    print("\n✅ Cálculo de volatilidades implícitas completado")


if __name__ == "__main__":
    main()