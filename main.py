"""
main.py
Script principal para ejecutar el proyecto completo de análisis de opciones
y entropía de distribuciones implícitas.

PROYECTO: Análisis de Opciones AAPL con Reconstrucción de Densidad Riesgo-Neutral
          y Cálculo de Entropías

Este script ejecuta todas las etapas del proyecto:
1. Descarga de datos de opciones y precios históricos
2. Preprocesamiento y limpieza de datos
3. Cálculo de volatilidades implícitas
4. Reconstrucción de densidades riesgo-neutral (RND)
5. Estimación de densidad física histórica
6. Cálculo de entropías (Shannon, Rényi) y divergencia KL
7. Simulación Monte Carlo de escenarios de estrés
8. Análisis comparativo y visualizaciones

EJECUCIÓN:
    python main.py [--skip-download] [--skip-plots]

ARGUMENTOS:
    --skip-download : Salta la descarga de datos (usa datos existentes)
    --skip-plots    : No genera gráficos (solo cálculos)

DEPENDENCIAS:
    Ver requirements.txt para todas las dependencias necesarias.
    Instalar con: pip install -r requirements.txt

SALIDAS:
    - data/          : Datos descargados y procesados
    - output/        : Resultados (CSV, pickle)
    - output/plots/  : Gráficos y visualizaciones
"""

import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Imports de módulos del proyecto
try:
    from src.data_loader import OptionsDataLoader
    from src.preprocessing import OptionsPreprocessor
    from src.implied_vol import ImpliedVolatilityCalculator
    from src.density_reconstruction import RiskNeutralDensity, PhysicalDensityEstimator
    from src.entropy_calculator import EntropyCalculator
    from src.monte_carlo import MonteCarloSimulator, create_stress_scenarios
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("\nAsegúrate de que todos los archivos estén en la carpeta src/")
    print("Ejecuta primero: python setup_project.py")
    sys.exit(1)

import pandas as pd
import numpy as np
import pickle


def print_header(title):
    """Imprime un encabezado formateado."""
    print(f"\n\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_section(title):
    """Imprime un separador de sección."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def print_step(step_num, total_steps, description):
    """Imprime el número de paso actual."""
    print(f"\n{'►'*3} PASO {step_num}/{total_steps}: {description} {'◄'*3}")


class OptionsAnalysisPipeline:
    """
    Pipeline completo para análisis de opciones y entropía.
    """
    
    def __init__(self, ticker='AAPL', skip_download=False, skip_plots=False):
        """
        Inicializa el pipeline.
        
        Args:
            ticker: Símbolo del activo
            skip_download: Si True, no descarga datos nuevos
            skip_plots: Si True, no genera gráficos
        """
        self.ticker = ticker
        self.skip_download = skip_download
        self.skip_plots = skip_plots
        
        # Configuración
        self.risk_free_rate = 0.045
        self.n_simulations = 10000
        self.historical_years = 3
        
        # Resultados intermedios
        self.df_options = None
        self.df_historical = None
        self.current_price = None
        self.historical_sigma = None
        
        print(f"\n{'🚀'*25}")
        print(f"{'PIPELINE DE ANÁLISIS DE OPCIONES':^50}")
        print(f"{'🚀'*25}")
        print(f"\n📊 Configuración:")
        print(f"  • Ticker: {self.ticker}")
        print(f"  • Tasa libre de riesgo: {self.risk_free_rate:.2%}")
        print(f"  • Simulaciones Monte Carlo: {self.n_simulations:,}")
        print(f"  • Años de datos históricos: {self.historical_years}")
        print(f"  • Descargar datos: {'No' if skip_download else 'Sí'}")
        print(f"  • Generar gráficos: {'No' if skip_plots else 'Sí'}")
    
    def run(self):
        """
        Ejecuta el pipeline completo.
        """
        start_time = time.time()
        
        try:
            # PASO 1: Descarga de datos
            print_step(1, 7, "DESCARGA DE DATOS")
            self.step1_download_data()
            
            # PASO 2: Preprocesamiento
            print_step(2, 7, "PREPROCESAMIENTO Y LIMPIEZA")
            self.step2_preprocessing()
            
            # PASO 3: Volatilidad implícita
            print_step(3, 7, "CÁLCULO DE VOLATILIDADES IMPLÍCITAS")
            self.step3_implied_volatility()
            
            # PASO 4: Reconstrucción de densidades
            print_step(4, 7, "RECONSTRUCCIÓN DE DENSIDADES")
            self.step4_density_reconstruction()
            
            # PASO 5: Cálculo de entropías
            print_step(5, 7, "CÁLCULO DE ENTROPÍAS")
            self.step5_entropy_calculation()
            
            # PASO 6: Simulación Monte Carlo
            print_step(6, 7, "SIMULACIÓN MONTE CARLO")
            self.step6_monte_carlo()
            
            # PASO 7: Análisis final y resumen
            print_step(7, 7, "ANÁLISIS FINAL Y RESUMEN")
            self.step7_final_analysis()
            
            # Tiempo total
            elapsed_time = time.time() - start_time
            
            print_header("✅ EJECUCIÓN COMPLETADA EXITOSAMENTE")
            print(f"⏱️  Tiempo total: {elapsed_time:.2f} segundos ({elapsed_time/60:.2f} minutos)")
            print(f"\n📁 Resultados guardados en:")
            print(f"  • Datos: data/")
            print(f"  • Resultados: output/")
            print(f"  • Gráficos: output/plots/")
            
            print(f"\n{'─'*70}")
            print("Para revisar los resultados:")
            print("  1. Revisa los archivos CSV en output/")
            print("  2. Revisa los gráficos en output/plots/")
            print("  3. Carga los archivos pickle para análisis adicional")
            print(f"{'─'*70}\n")
            
        except Exception as e:
            print(f"\n\n❌ ERROR EN LA EJECUCIÓN: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def step1_download_data(self):
        """Paso 1: Descarga datos de opciones e históricos."""
        if self.skip_download and (Path('data') / f'{self.ticker}_options.csv').exists():
            print("\n⏭️  Saltando descarga - usando datos existentes")
            
            # Cargar datos existentes
            self.df_options = pd.read_csv(f'data/{self.ticker}_options.csv')
            self.df_options['expiration'] = pd.to_datetime(self.df_options['expiration'])
            self.current_price = self.df_options['stockPrice'].iloc[0]
            
            self.df_historical = pd.read_csv(f'data/{self.ticker}_historical.csv', 
                                            index_col=0, parse_dates=True)
            
            print(f"✓ Datos cargados: {len(self.df_options)} opciones")
            print(f"✓ Precio actual: ${self.current_price:.2f}")
        else:
            print("\n📥 Descargando datos de Yahoo Finance...")
            
            loader = OptionsDataLoader(ticker=self.ticker, data_dir='data')
            
            # Descargar opciones
            self.df_options = loader.download_options_data()
            self.current_price = loader.current_price
            
            # Descargar históricos
            self.df_historical = loader.download_historical_data(years=self.historical_years)
        
        # Calcular volatilidad histórica
        if 'log_return' not in self.df_historical.columns:
            self.df_historical['log_return'] = np.log(
                self.df_historical['Close'] / self.df_historical['Close'].shift(1)
            )
        
        self.historical_sigma = self.df_historical['log_return'].std() * np.sqrt(252)
        
        print(f"\n✅ Paso 1 completado")
        print(f"   • Opciones descargadas: {len(self.df_options)}")
        print(f"   • Días históricos: {len(self.df_historical)}")
        print(f"   • Volatilidad histórica: {self.historical_sigma:.2%}")
    
    def step2_preprocessing(self):
        """Paso 2: Preprocesamiento y limpieza."""
        print("\n🧹 Limpiando y validando datos...")
        
        preprocessor = OptionsPreprocessor(min_price=0.01, min_volume=0, min_oi=0)
        
        self.df_options = preprocessor.clean_options_data(self.df_options)
        preprocessor.save_clean_data(self.df_options, ticker=self.ticker, data_dir='data')
        
        print(f"\n✅ Paso 2 completado")
        print(f"   • Opciones válidas: {len(self.df_options)}")
    
    def step3_implied_volatility(self):
        """Paso 3: Cálculo de volatilidades implícitas."""
        print("\n📈 Calculando volatilidades implícitas...")
        
        iv_calculator = ImpliedVolatilityCalculator(risk_free_rate=self.risk_free_rate)
        
        self.df_options = iv_calculator.calculate_implied_volatilities(self.df_options)
        
        # Guardar
        output_file = Path('data') / f'{self.ticker}_options_with_iv.csv'
        self.df_options.to_csv(output_file, index=False)
        
        # Graficar
        if not self.skip_plots:
            iv_calculator.plot_volatility_smile(self.df_options)
        
        print(f"\n✅ Paso 3 completado")
        print(f"   • IVs calculadas para {len(self.df_options)} opciones")
    
    def step4_density_reconstruction(self):
        """Paso 4: Reconstrucción de densidades RND y física."""
        print("\n📊 Reconstruyendo densidades...")
        
        # Densidad riesgo-neutral
        rnd = RiskNeutralDensity(
            risk_free_rate=self.risk_free_rate,
            smoothing=0.01,
            n_grid=500
        )
        
        self.densities_rnd = rnd.reconstruct_all_maturities(self.df_options)
        rnd.save_densities(output_dir='output')
        
        if not self.skip_plots:
            rnd.plot_densities(output_dir='output/plots')
        
        # Densidad física
        phys_estimator = PhysicalDensityEstimator(bandwidth='scott')
        self.density_phys = phys_estimator.estimate_from_returns(
            self.df_historical,
            self.current_price,
            horizon_days=30
        )
        
        phys_estimator.save_physical_density(output_dir='output')
        
        if not self.skip_plots:
            phys_estimator.plot_physical_density(output_dir='output/plots')
        
        print(f"\n✅ Paso 4 completado")
        print(f"   • Densidades RND: {len(self.densities_rnd)} vencimientos")
        print(f"   • Densidad física: {'✓' if self.density_phys else '✗'}")
    
    def step5_entropy_calculation(self):
        """Paso 5: Cálculo de entropías y primas de riesgo."""
        print("\n📐 Calculando entropías...")
        
        entropy_calc = EntropyCalculator()
        
        # Calcular entropías
        self.df_entropy = entropy_calc.calculate_all_entropies(
            self.densities_rnd,
            self.density_phys,
            renyi_alphas=[0.5, 1.5, 2.0]
        )
        
        # Primas de riesgo
        self.df_erp = entropy_calc.calculate_entropy_risk_premium(self.df_entropy)
        
        # Guardar
        entropy_calc.save_entropy_results(self.df_entropy, self.df_erp, output_dir='output')
        
        # Graficar
        if not self.skip_plots:
            entropy_calc.plot_entropy_analysis(self.df_entropy, self.df_erp, 
                                              output_dir='output/plots')
        
        print(f"\n✅ Paso 5 completado")
        print(f"   • Entropías calculadas: {len(self.df_entropy)} registros")
    
    def step6_monte_carlo(self):
        """Paso 6: Simulación Monte Carlo de escenarios de estrés."""
        print("\n🎲 Ejecutando simulaciones Monte Carlo...")
        
        simulator = MonteCarloSimulator(
            S0=self.current_price,
            r=self.risk_free_rate,
            n_simulations=self.n_simulations,
            seed=42
        )
        
        # Configurar escenarios
        base_params = {'sigma': self.historical_sigma}
        stress_scenarios = create_stress_scenarios(self.historical_sigma)
        
        # Simular (30 días)
        T = 30 / 365.0
        self.mc_results = simulator.run_stress_scenarios(T, base_params, stress_scenarios)
        
        # Calcular entropías de escenarios
        self.df_scenario_entropy = simulator.calculate_scenario_entropies()
        
        # Guardar
        simulator.save_results(output_dir='output')
        
        entropy_file = Path('output') / 'scenario_entropies.csv'
        self.df_scenario_entropy.to_csv(entropy_file, index=False)
        
        # Graficar
        if not self.skip_plots:
            simulator.plot_simulation_results(output_dir='output/plots')
        
        print(f"\n✅ Paso 6 completado")
        print(f"   • Escenarios simulados: {len(self.mc_results)}")
    
    def step7_final_analysis(self):
        """Paso 7: Análisis final y resumen."""
        print("\n📊 Generando análisis final...")
        
        # Crear resumen ejecutivo
        summary = []
        
        summary.append(f"{'='*70}")
        summary.append(f"{'RESUMEN EJECUTIVO - ANÁLISIS DE OPCIONES':^70}")
        summary.append(f"{'='*70}\n")
        
        summary.append(f"Ticker: {self.ticker}")
        summary.append(f"Precio actual: ${self.current_price:.2f}")
        summary.append(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        summary.append(f"{'─'*70}")
        summary.append("DATOS PROCESADOS")
        summary.append(f"{'─'*70}")
        summary.append(f"  • Opciones analizadas: {len(self.df_options)}")
        summary.append(f"  • Vencimientos únicos: {self.df_options['expiration'].nunique()}")
        summary.append(f"  • Días históricos: {len(self.df_historical)}")
        summary.append(f"  • Volatilidad histórica: {self.historical_sigma:.2%}\n")
        
        summary.append(f"{'─'*70}")
        summary.append("DENSIDADES RECONSTRUIDAS")
        summary.append(f"{'─'*70}")
        summary.append(f"  • Densidades RND exitosas: {len(self.densities_rnd)}")
        summary.append(f"  • Densidad física: {'✓ Estimada' if self.density_phys else '✗ No disponible'}\n")
        
        summary.append(f"{'─'*70}")
        summary.append("ENTROPÍAS")
        summary.append(f"{'─'*70}")
        if not self.df_entropy.empty:
            avg_shannon = self.df_entropy['H_shannon'].mean()
            min_shannon = self.df_entropy['H_shannon'].min()
            max_shannon = self.df_entropy['H_shannon'].max()
            
            summary.append(f"  • Entropía Shannon promedio: {avg_shannon:.4f} nats")
            summary.append(f"  • Rango de entropía: [{min_shannon:.4f}, {max_shannon:.4f}]")
            
            if 'KL_vs_physical' in self.df_entropy.columns:
                kl_valid = self.df_entropy['KL_vs_physical'].dropna()
                if len(kl_valid) > 0:
                    avg_kl = kl_valid.mean()
                    summary.append(f"  • KL divergence promedio (RND||Física): {avg_kl:.4f} nats")
        summary.append("")
        
        summary.append(f"{'─'*70}")
        summary.append("SIMULACIÓN MONTE CARLO")
        summary.append(f"{'─'*70}")
        summary.append(f"  • Escenarios simulados: {len(self.mc_results)}")
        summary.append(f"  • Simulaciones por escenario: {self.n_simulations:,}")
        
        # Estadísticas de escenarios
        if hasattr(self, 'mc_results'):
            for scenario_name, data in self.mc_results.items():
                summary.append(f"\n  {data['scenario']}:")
                summary.append(f"    - Precio medio: ${data['mean_price']:.2f}")
                summary.append(f"    - Retorno esperado: {data['mean_return']:.2%}")
                summary.append(f"    - Probabilidad pérdida: {data['prob_loss']:.2%}")
                summary.append(f"    - VaR(95%): ${data['var_95']:.2f}")
        
        summary.append(f"\n{'─'*70}")
        summary.append("ARCHIVOS GENERADOS")
        summary.append(f"{'─'*70}")
        summary.append("  Datos:")
        summary.append(f"    • data/{self.ticker}_options.csv")
        summary.append(f"    • data/{self.ticker}_options_clean.csv")
        summary.append(f"    • data/{self.ticker}_options_with_iv.csv")
        summary.append(f"    • data/{self.ticker}_historical.csv")
        summary.append("")
        summary.append("  Resultados:")
        summary.append("    • output/rnd_densities.pkl")
        summary.append("    • output/rnd_summary.csv")
        summary.append("    • output/density_physical.pkl")
        summary.append("    • output/entropy_by_maturity.csv")
        summary.append("    • output/entropy_risk_premium.csv")
        summary.append("    • output/monte_carlo_results.pkl")
        summary.append("    • output/monte_carlo_summary.csv")
        summary.append("    • output/scenario_entropies.csv")
        
        if not self.skip_plots:
            summary.append("")
            summary.append("  Gráficos:")
            summary.append("    • output/plots/volatility_smile.png")
            summary.append("    • output/plots/iv_by_strike.png")
            summary.append("    • output/plots/rnd_densities.png")
            summary.append("    • output/plots/rnd_cdf.png")
            summary.append("    • output/plots/density_physical.png")
            summary.append("    • output/plots/entropy_analysis.png")
            summary.append("    • output/plots/entropy_risk_premium.png")
            summary.append("    • output/plots/mc_distributions.png")
            summary.append("    • output/plots/mc_statistics.png")
            summary.append("    • output/plots/mc_risk_metrics.png")
        
        summary.append(f"\n{'='*70}\n")
        
        # Imprimir resumen
        summary_text = '\n'.join(summary)
        print(summary_text)
        
        # Guardar resumen
        summary_file = Path('output') / 'executive_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"💾 Resumen ejecutivo guardado en: {summary_file}")
        
        print(f"\n✅ Paso 7 completado")


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    skip_download = '--skip-download' in sys.argv
    skip_plots = '--skip-plots' in sys.argv
    
    return skip_download, skip_plots


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 
        'yfinance', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Faltan paquetes requeridos: {', '.join(missing)}")
        print("\nInstala las dependencias con:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def main():
    """
    Función principal.
    """
    # Banner inicial
    print("\n" + "="*70)
    print("  ANÁLISIS DE OPCIONES Y ENTROPÍA DE DISTRIBUCIONES IMPLÍCITAS")
    print("  Ticker: AAPL (Apple Inc.)")
    print("="*70)
    
    # Verificar dependencias
    print("\n🔍 Verificando dependencias...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Todas las dependencias instaladas")
    
    # Verificar estructura de proyecto
    required_dirs = ['data', 'output', 'output/plots', 'src']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Parsear argumentos
    skip_download, skip_plots = parse_arguments()
    
    if skip_download:
        print("\n⚠️  Modo: Usando datos existentes (sin descargar)")
    if skip_plots:
        print("⚠️  Modo: Sin generar gráficos")
    
    # Crear y ejecutar pipeline
    pipeline = OptionsAnalysisPipeline(
        ticker='AAPL',
        skip_download=skip_download,
        skip_plots=skip_plots
    )
    
    pipeline.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)