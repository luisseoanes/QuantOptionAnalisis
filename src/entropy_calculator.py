"""
entropy_calculator.py
Módulo para calcular entropías (Shannon, Rényi) y divergencia KL.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class EntropyCalculator:
    """
    Calcula medidas de entropía e incertidumbre para distribuciones.
    """
    
    def __init__(self):
        """Inicializa el calculador de entropía."""
        self.entropy_results = []
    
    @staticmethod
    def shannon_entropy(density, x_grid):
        """
        Calcula entropía de Shannon (diferencial).
        
        H = -∫ p(x) log(p(x)) dx
        
        Args:
            density: Array con valores de densidad
            x_grid: Array con grilla de valores
            
        Returns:
            Entropía de Shannon (en nats)
        """
        # Evitar log(0)
        density_safe = np.maximum(density, 1e-10)
        
        # Calcular entropía
        integrand = density * np.log(density_safe)
        H = -np.trapz(integrand, x_grid)
        
        return H
    
    @staticmethod
    def renyi_entropy(density, x_grid, alpha):
        """
        Calcula entropía de Rényi de orden alpha.
        
        H_α = (1/(1-α)) * log(∫ p(x)^α dx)
        
        Args:
            density: Array con valores de densidad
            x_grid: Array con grilla de valores
            alpha: Orden de la entropía (α ≠ 1)
            
        Returns:
            Entropía de Rényi
        """
        if alpha == 1.0:
            return EntropyCalculator.shannon_entropy(density, x_grid)
        
        # Calcular integral de p^α
        density_safe = np.maximum(density, 1e-10)
        integrand = density_safe ** alpha
        integral = np.trapz(integrand, x_grid)
        
        # Entropía de Rényi
        if integral <= 0:
            return np.nan
        
        H_alpha = (1.0 / (1.0 - alpha)) * np.log(integral)
        
        return H_alpha
    
    @staticmethod
    def kl_divergence(p, q, x_grid):
        """
        Calcula divergencia de Kullback-Leibler entre dos distribuciones.
        
        KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx
        
        Args:
            p: Densidad P (típicamente la "verdadera")
            q: Densidad Q (típicamente la aproximación)
            x_grid: Grilla de valores
            
        Returns:
            Divergencia KL (en nats)
        """
        # Asegurar que ambas tienen soporte positivo
        p_safe = np.maximum(p, 1e-10)
        q_safe = np.maximum(q, 1e-10)
        
        # Calcular KL
        integrand = p * np.log(p_safe / q_safe)
        kl = np.trapz(integrand, x_grid)
        
        return kl
    
    def calculate_all_entropies(self, densities_rnd, density_phys=None, 
                               renyi_alphas=[0.5, 1.5, 2.0]):
        """
        Calcula todas las entropías para densidades RND.
        
        Args:
            densities_rnd: dict con densidades riesgo-neutral por vencimiento
            density_phys: dict con densidad física (opcional)
            renyi_alphas: Lista de valores de alpha para Rényi
            
        Returns:
            DataFrame con resultados
        """
        print(f"\n{'='*60}")
        print("CALCULANDO ENTROPÍAS")
        print(f"{'='*60}")
        
        results = []
        
        for exp, methods in densities_rnd.items():
            print(f"\n{'─'*60}")
            print(f"Vencimiento: {exp.date()}")
            
            for method_name, rnd in methods.items():
                if rnd is None:
                    continue
                
                print(f"\n  Método: {method_name}")
                
                strikes = rnd['strikes']
                density = rnd['density']
                T_days = int(rnd['T'] * 365)
                
                # Shannon entropy
                H_shannon = self.shannon_entropy(density, strikes)
                print(f"    • Shannon entropy: {H_shannon:.4f} nats")
                
                # Rényi entropies
                H_renyi = {}
                for alpha in renyi_alphas:
                    H_alpha = self.renyi_entropy(density, strikes, alpha)
                    H_renyi[alpha] = H_alpha
                    print(f"    • Rényi entropy (α={alpha}): {H_alpha:.4f} nats")
                
                # KL divergence vs física (si está disponible)
                kl_vs_phys = np.nan
                if density_phys is not None:
                    kl_vs_phys = self._calculate_kl_vs_physical(
                        rnd, density_phys
                    )
                    if not np.isnan(kl_vs_phys):
                        print(f"    • KL(RND||Physical): {kl_vs_phys:.4f} nats")
                
                # Guardar resultados
                result = {
                    'expiration': exp,
                    'T_days': T_days,
                    'T_years': rnd['T'],
                    'method': method_name,
                    'H_shannon': H_shannon,
                    'KL_vs_physical': kl_vs_phys,
                    'mean': rnd['mean'],
                    'std': rnd['std'],
                    'S_current': rnd['S']
                }
                
                # Añadir Rényi entropies
                for alpha in renyi_alphas:
                    result[f'H_renyi_alpha{alpha:.1f}'.replace('.', '')] = H_renyi[alpha]
                
                results.append(result)
        
        # Crear DataFrame
        df_entropy = pd.DataFrame(results)
        self.entropy_results = df_entropy
        
        print(f"\n{'='*60}")
        print("RESUMEN DE ENTROPÍAS")
        print(f"{'='*60}")
        print(f"\nCálculos completados: {len(df_entropy)}")
        print(f"\nEstadísticas de Shannon entropy:")
        print(df_entropy.groupby('method')['H_shannon'].describe())
        
        return df_entropy
    
    def _calculate_kl_vs_physical(self, rnd, density_phys):
        """
        Calcula KL entre RND y densidad física.
        
        Args:
            rnd: dict con densidad RND
            density_phys: dict con densidad física
            
        Returns:
            Divergencia KL
        """
        try:
            # Interpolar densidades a una grilla común
            strikes_rnd = rnd['strikes']
            density_rnd_vals = rnd['density']
            
            strikes_phys = density_phys['strikes']
            density_phys_vals = density_phys['density']
            
            # Encontrar rango común
            k_min = max(strikes_rnd.min(), strikes_phys.min())
            k_max = min(strikes_rnd.max(), strikes_phys.max())
            
            # Crear grilla común
            k_common = np.linspace(k_min, k_max, 300)
            
            # Interpolar
            interp_rnd = interp1d(strikes_rnd, density_rnd_vals, 
                                 kind='linear', fill_value=0, bounds_error=False)
            interp_phys = interp1d(strikes_phys, density_phys_vals, 
                                  kind='linear', fill_value=0, bounds_error=False)
            
            density_rnd_interp = interp_rnd(k_common)
            density_phys_interp = interp_phys(k_common)
            
            # Normalizar
            density_rnd_interp = density_rnd_interp / np.trapz(density_rnd_interp, k_common)
            density_phys_interp = density_phys_interp / np.trapz(density_phys_interp, k_common)
            
            # Calcular KL(RND||Physical)
            kl = self.kl_divergence(density_rnd_interp, density_phys_interp, k_common)
            
            return kl
            
        except Exception as e:
            print(f"      ⚠️ Error calculando KL: {str(e)}")
            return np.nan
    
    def calculate_entropy_risk_premium(self, df_entropy):
        """
        Calcula prima de riesgo de entropía.
        
        ERP = diferencia entre entropía RND y física
        
        Args:
            df_entropy: DataFrame con entropías calculadas
            
        Returns:
            DataFrame con primas de riesgo
        """
        print(f"\n{'='*60}")
        print("CALCULANDO PRIMAS DE RIESGO DE ENTROPÍA")
        print(f"{'='*60}")
        
        # Agrupar por vencimiento
        results = []
        
        for exp in df_entropy['expiration'].unique():
            df_exp = df_entropy[df_entropy['expiration'] == exp]
            
            if len(df_exp) == 0:
                continue
            
            T_days = df_exp['T_days'].iloc[0]
            
            # Promedio de entropías RND (entre métodos)
            H_rnd_mean = df_exp['H_shannon'].mean()
            
            # KL divergence (si está disponible)
            kl_values = df_exp['KL_vs_physical'].dropna()
            kl_mean = kl_values.mean() if len(kl_values) > 0 else np.nan
            
            results.append({
                'expiration': exp,
                'T_days': T_days,
                'H_rnd_mean': H_rnd_mean,
                'KL_mean': kl_mean,
                'entropy_risk_premium': kl_mean  # Proxy simple
            })
        
        df_erp = pd.DataFrame(results)
        
        print(f"\n✅ Primas de riesgo calculadas para {len(df_erp)} vencimientos")
        
        if not df_erp.empty:
            print(f"\nEstadísticas:")
            print(df_erp[['T_days', 'H_rnd_mean', 'KL_mean']].describe())
        
        return df_erp
    
    def save_entropy_results(self, df_entropy, df_erp=None, output_dir='output'):
        """
        Guarda resultados de entropía.
        
        Args:
            df_entropy: DataFrame con entropías
            df_erp: DataFrame con primas de riesgo (opcional)
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar entropías por vencimiento
        entropy_file = output_path / 'entropy_by_maturity.csv'
        df_entropy.to_csv(entropy_file, index=False)
        print(f"\n💾 Entropías guardadas en: {entropy_file}")
        
        # Guardar primas de riesgo
        if df_erp is not None:
            erp_file = output_path / 'entropy_risk_premium.csv'
            df_erp.to_csv(erp_file, index=False)
            print(f"💾 Primas de riesgo guardadas en: {erp_file}")
    
    def plot_entropy_analysis(self, df_entropy, df_erp=None, output_dir='output/plots'):
        """
        Grafica análisis de entropía.
        
        Args:
            df_entropy: DataFrame con entropías
            df_erp: DataFrame con primas de riesgo (opcional)
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GRAFICANDO ANÁLISIS DE ENTROPÍA")
        print(f"{'='*60}")
        
        # Figura 1: Entropía de Shannon vs tiempo a vencimiento
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Shannon entropy por método
        ax1 = axes[0, 0]
        for method in df_entropy['method'].unique():
            df_method = df_entropy[df_entropy['method'] == method]
            ax1.plot(df_method['T_days'], df_method['H_shannon'], 
                    'o-', label=method, markersize=8, linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Días hasta vencimiento', fontsize=11)
        ax1.set_ylabel('Shannon Entropy (nats)', fontsize=11)
        ax1.set_title('Shannon Entropy vs Madurez', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rényi entropies (alpha=0.5, 1.5, 2.0)
        ax2 = axes[0, 1]
        renyi_cols = [col for col in df_entropy.columns if 'H_renyi' in col]
        
        if renyi_cols:
            # Promediar entre métodos para cada vencimiento
            df_avg = df_entropy.groupby('T_days')[renyi_cols + ['H_shannon']].mean().reset_index()
            
            for col in renyi_cols:
                alpha_str = col.replace('H_renyi_alpha', '').replace('0', '.0')
                ax2.plot(df_avg['T_days'], df_avg[col], 
                        'o-', label=f'α={alpha_str}', markersize=7, linewidth=2, alpha=0.7)
            
            ax2.plot(df_avg['T_days'], df_avg['H_shannon'], 
                    's-', label='Shannon (α=1)', markersize=7, linewidth=2, 
                    alpha=0.7, color='black')
        
        ax2.set_xlabel('Días hasta vencimiento', fontsize=11)
        ax2.set_ylabel('Rényi Entropy (nats)', fontsize=11)
        ax2.set_title('Rényi Entropy vs Madurez', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # KL divergence vs física
        ax3 = axes[1, 0]
        df_kl = df_entropy[df_entropy['KL_vs_physical'].notna()]
        
        if len(df_kl) > 0:
            for method in df_kl['method'].unique():
                df_method = df_kl[df_kl['method'] == method]
                ax3.plot(df_method['T_days'], df_method['KL_vs_physical'], 
                        'o-', label=method, markersize=8, linewidth=2, alpha=0.7)
            
            ax3.set_xlabel('Días hasta vencimiento', fontsize=11)
            ax3.set_ylabel('KL Divergence (nats)', fontsize=11)
            ax3.set_title('KL(RND||Physical) vs Madurez', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No hay datos de KL disponibles', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('KL Divergence', fontsize=12, fontweight='bold')
        
        # Relación entre entropía y volatilidad implícita
        ax4 = axes[1, 1]
        df_avg_entropy = df_entropy.groupby('T_days').agg({
            'H_shannon': 'mean',
            'std': 'mean'
        }).reset_index()
        
        ax4.scatter(df_avg_entropy['std'], df_avg_entropy['H_shannon'], 
                   s=100, alpha=0.6, c=df_avg_entropy['T_days'], cmap='viridis')
        
        # Añadir colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Días a vencimiento', fontsize=10)
        
        ax4.set_xlabel('Std Dev RND ($)', fontsize=11)
        ax4.set_ylabel('Shannon Entropy (nats)', fontsize=11)
        ax4.set_title('Entropía vs Incertidumbre', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Análisis de Entropía - AAPL Options', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_path / 'entropy_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_file}")
        
        plt.close()
        
        # Figura 2: Primas de riesgo (si están disponibles)
        if df_erp is not None and not df_erp.empty:
            self._plot_risk_premium(df_erp, output_path)
    
    def _plot_risk_premium(self, df_erp, output_path):
        """
        Grafica primas de riesgo de entropía.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prima de riesgo vs madurez
        df_valid = df_erp[df_erp['KL_mean'].notna()]
        
        if len(df_valid) > 0:
            ax1.plot(df_valid['T_days'], df_valid['KL_mean'], 
                    'o-', color='red', markersize=10, linewidth=2.5, alpha=0.7)
            ax1.fill_between(df_valid['T_days'], 0, df_valid['KL_mean'], 
                            alpha=0.2, color='red')
            ax1.set_xlabel('Días hasta vencimiento', fontsize=12)
            ax1.set_ylabel('Entropy Risk Premium (nats)', fontsize=12)
            ax1.set_title('Prima de Riesgo de Entropía', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No hay datos disponibles', 
                    ha='center', va='center', fontsize=12)
        
        # Entropía RND promedio vs madurez
        ax2.plot(df_erp['T_days'], df_erp['H_rnd_mean'], 
                'o-', color='blue', markersize=10, linewidth=2.5, alpha=0.7)
        ax2.set_xlabel('Días hasta vencimiento', fontsize=12)
        ax2.set_ylabel('Shannon Entropy (nats)', fontsize=12)
        ax2.set_title('Entropía RND Promedio', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Primas de Riesgo y Entropía - AAPL', 
                    fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_file = output_path / 'entropy_risk_premium.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {output_file}")
        
        plt.close()


def main():
    """
    Función principal para cálculo de entropías.
    """
    from pathlib import Path
    
    # Cargar densidades RND
    rnd_file = Path('output') / 'rnd_densities.pkl'
    phys_file = Path('output') / 'density_physical.pkl'
    
    if not rnd_file.exists():
        print(f"❌ Error: No se encuentra {rnd_file}")
        print("   Ejecuta primero density_reconstruction.py")
        return
    
    print(f"📂 Cargando densidades RND desde {rnd_file}")
    with open(rnd_file, 'rb') as f:
        densities_rnd = pickle.load(f)
    
    # Cargar densidad física (si existe)
    density_phys = None
    if phys_file.exists():
        print(f"📂 Cargando densidad física desde {phys_file}")
        with open(phys_file, 'rb') as f:
            density_phys = pickle.load(f)
    else:
        print("⚠️  No se encontró densidad física")
    
    # Crear calculador
    entropy_calc = EntropyCalculator()
    
    # Calcular entropías
    df_entropy = entropy_calc.calculate_all_entropies(
        densities_rnd, 
        density_phys,
        renyi_alphas=[0.5, 1.5, 2.0]
    )
    
    # Calcular primas de riesgo
    df_erp = entropy_calc.calculate_entropy_risk_premium(df_entropy)
    
    # Guardar resultados
    entropy_calc.save_entropy_results(df_entropy, df_erp, output_dir='output')
    
    # Graficar
    entropy_calc.plot_entropy_analysis(df_entropy, df_erp, output_dir='output/plots')
    
    print("\n✅ Cálculo de entropías completado")


if __name__ == "__main__":
    main()