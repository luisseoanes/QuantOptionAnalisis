"""
data_loader.py
Módulo para descargar datos de opciones y precios históricos usando yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OptionsDataLoader:
    """
    Clase para descargar y gestionar datos de opciones de Yahoo Finance.
    """
    
    def __init__(self, ticker='AAPL', data_dir='data'):
        """
        Inicializa el cargador de datos.
        
        Args:
            ticker: Símbolo del activo
            data_dir: Directorio para guardar datos
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.stock = yf.Ticker(ticker)
        self.current_price = None
        
    def download_options_data(self):
        """
        Descarga datos de opciones para todas las fechas de vencimiento disponibles.
        
        Returns:
            DataFrame con todas las opciones (calls y puts)
        """
        print(f"\n{'='*60}")
        print(f"DESCARGANDO DATOS DE OPCIONES PARA {self.ticker}")
        print(f"{'='*60}")
        
        # Obtener precio actual
        hist = self.stock.history(period='1d')
        self.current_price = hist['Close'].iloc[-1]
        print(f"\n📊 Precio actual de {self.ticker}: ${self.current_price:.2f}")
        
        # Obtener fechas de vencimiento disponibles
        expirations = self.stock.options
        print(f"📅 Fechas de vencimiento disponibles: {len(expirations)}")
        
        if len(expirations) == 0:
            raise ValueError(f"No hay opciones disponibles para {self.ticker}")
        
        all_options = []
        
        for exp_date in expirations:
            try:
                # Obtener cadena de opciones para esta fecha
                opt_chain = self.stock.option_chain(exp_date)
                
                # Procesar calls
                calls = opt_chain.calls.copy()
                calls['type'] = 'call'
                calls['expiration'] = exp_date
                
                # Procesar puts
                puts = opt_chain.puts.copy()
                puts['type'] = 'put'
                puts['expiration'] = exp_date
                
                # Combinar
                options = pd.concat([calls, puts], ignore_index=True)
                all_options.append(options)
                
                print(f"  ✓ {exp_date}: {len(calls)} calls, {len(puts)} puts")
                
            except Exception as e:
                print(f"  ✗ Error en {exp_date}: {str(e)}")
                continue
        
        if not all_options:
            raise ValueError("No se pudieron descargar opciones")
        
        # Combinar todas las opciones
        df_options = pd.concat(all_options, ignore_index=True)
        
        # Añadir información adicional
        df_options['stockPrice'] = self.current_price
        df_options['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Calcular días hasta vencimiento
        df_options['expiration'] = pd.to_datetime(df_options['expiration'])
        df_options['T_days'] = (df_options['expiration'] - pd.Timestamp.now()).dt.days
        df_options['T_years'] = df_options['T_days'] / 365.0
        
        # Calcular moneyness
        df_options['moneyness'] = df_options['strike'] / self.current_price
        
        print(f"\n✅ Total de opciones descargadas: {len(df_options)}")
        
        # Guardar archivos separados
        self._save_options_data(df_options)
        
        return df_options
    
    def _save_options_data(self, df_options):
        """
        Guarda los datos de opciones en archivos CSV separados.
        
        Args:
            df_options: DataFrame con todas las opciones
        """
        # Archivo completo
        file_all = self.data_dir / f'{self.ticker}_options.csv'
        df_options.to_csv(file_all, index=False)
        print(f"\n💾 Guardado: {file_all}")
        
        # Calls separados
        df_calls = df_options[df_options['type'] == 'call']
        file_calls = self.data_dir / f'{self.ticker}_calls.csv'
        df_calls.to_csv(file_calls, index=False)
        print(f"💾 Guardado: {file_calls} ({len(df_calls)} calls)")
        
        # Puts separados
        df_puts = df_options[df_options['type'] == 'put']
        file_puts = self.data_dir / f'{self.ticker}_puts.csv'
        df_puts.to_csv(file_puts, index=False)
        print(f"💾 Guardado: {file_puts} ({len(df_puts)} puts)")
    
    def download_historical_data(self, years=3):
        """
        Descarga datos históricos de precios.
        
        Args:
            years: Años de historia a descargar
            
        Returns:
            DataFrame con precios históricos
        """
        print(f"\n{'='*60}")
        print(f"DESCARGANDO DATOS HISTÓRICOS DE {self.ticker}")
        print(f"{'='*60}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"\n📅 Período: {start_date.date()} a {end_date.date()}")
        
        # Descargar datos
        df_hist = self.stock.history(start=start_date, end=end_date)
        
        if df_hist.empty:
            raise ValueError(f"No se pudieron descargar datos históricos para {self.ticker}")
        
        # Calcular retornos logarítmicos
        df_hist['log_return'] = np.log(df_hist['Close'] / df_hist['Close'].shift(1))
        df_hist['simple_return'] = df_hist['Close'].pct_change()
        
        # Guardar
        file_hist = self.data_dir / f'{self.ticker}_historical.csv'
        df_hist.to_csv(file_hist)
        print(f"\n💾 Guardado: {file_hist}")
        print(f"📊 {len(df_hist)} días de datos históricos")
        print(f"📈 Retorno promedio diario: {df_hist['log_return'].mean():.4%}")
        print(f"📉 Volatilidad diaria: {df_hist['log_return'].std():.4%}")
        print(f"📉 Volatilidad anualizada: {df_hist['log_return'].std() * np.sqrt(252):.4%}")
        
        return df_hist
    
    def load_saved_data(self, data_type='options'):
        """
        Carga datos guardados previamente.
        
        Args:
            data_type: 'options', 'calls', 'puts', o 'historical'
            
        Returns:
            DataFrame con los datos
        """
        file_mapping = {
            'options': f'{self.ticker}_options.csv',
            'calls': f'{self.ticker}_calls.csv',
            'puts': f'{self.ticker}_puts.csv',
            'historical': f'{self.ticker}_historical.csv'
        }
        
        if data_type not in file_mapping:
            raise ValueError(f"data_type debe ser uno de: {list(file_mapping.keys())}")
        
        file_path = self.data_dir / file_mapping[data_type]
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Convertir fechas si es necesario
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
        
        if data_type == 'historical':
            df['Date'] = pd.to_datetime(df['Date']) if 'Date' in df.columns else df.index
            df = df.set_index('Date')
        
        print(f"✓ Cargados {len(df)} registros de {file_path.name}")
        
        return df


def main():
    """
    Función principal para ejecutar la descarga de datos.
    """
    # Crear instancia del cargador
    loader = OptionsDataLoader(ticker='AAPL', data_dir='data')
    
    # Descargar datos de opciones
    df_options = loader.download_options_data()
    
    # Descargar datos históricos (3 años)
    df_historical = loader.download_historical_data(years=3)
    
    print(f"\n{'='*60}")
    print("✅ DESCARGA COMPLETADA")
    print(f"{'='*60}")
    print(f"\nResumen:")
    print(f"  • Opciones totales: {len(df_options)}")
    print(f"  • Vencimientos únicos: {df_options['expiration'].nunique()}")
    print(f"  • Días históricos: {len(df_historical)}")
    print(f"\nArchivos guardados en: data/")


if __name__ == "__main__":
    main()