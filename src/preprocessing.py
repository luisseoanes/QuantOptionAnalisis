"""
preprocessing.py
Módulo para limpiar y validar datos de opciones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OptionsPreprocessor:
    """
    Clase para preprocesar y limpiar datos de opciones.
    """
    
    def __init__(self, min_price=0.01, min_volume=0, min_oi=0):
        """
        Inicializa el preprocesador.
        
        Args:
            min_price: Precio mínimo válido
            min_volume: Volumen mínimo
            min_oi: Open Interest mínimo
        """
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_oi = min_oi
        
    def clean_options_data(self, df_options):
        """
        Limpia y valida datos de opciones.
        
        Args:
            df_options: DataFrame con datos de opciones
            
        Returns:
            DataFrame limpio
        """
        print(f"\n{'='*60}")
        print("PREPROCESAMIENTO Y LIMPIEZA DE DATOS")
        print(f"{'='*60}")
        
        df = df_options.copy()
        n_initial = len(df)
        print(f"\n📊 Registros iniciales: {n_initial}")
        
        # 1. Eliminar filas con precios inválidos
        print("\n1️⃣ Limpiando precios inválidos...")
        
        # Verificar columnas disponibles
        price_cols = []
        for col in ['lastPrice', 'bid', 'ask']:
            if col in df.columns:
                price_cols.append(col)
        
        if not price_cols:
            print("⚠️  No se encontraron columnas de precio")
        else:
            # Eliminar donde todos los precios son <= 0
            mask_valid = pd.Series(True, index=df.index)
            for col in price_cols:
                mask_valid &= (df[col] > self.min_price) | df[col].isna()
            
            # Al menos un precio debe ser válido
            mask_any_valid = pd.Series(False, index=df.index)
            for col in price_cols:
                mask_any_valid |= (df[col] > self.min_price)
            
            df = df[mask_any_valid].copy()
            print(f"  ✓ Eliminados {n_initial - len(df)} registros con precios inválidos")
        
        # 2. Filtrar por volumen y open interest
        print("\n2️⃣ Filtrando por liquidez...")
        
        n_before = len(df)
        
        if 'volume' in df.columns and 'openInterest' in df.columns:
            # Mantener opciones con volumen > 0 O openInterest > 0
            df = df[
                (df['volume'] > self.min_volume) | 
                (df['openInterest'] > self.min_oi)
            ].copy()
            print(f"  ✓ Filtrados {n_before - len(df)} registros por liquidez")
        else:
            print("  ℹ️  Columnas de liquidez no disponibles, saltando filtro")
        
        # 3. Verificar y crear columna de precio mid
        print("\n3️⃣ Calculando precio medio (mid)...")
        
        if 'bid' in df.columns and 'ask' in df.columns:
            # Usar mid donde sea posible
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            
            # Si mid es inválido, usar lastPrice
            if 'lastPrice' in df.columns:
                mask_invalid_mid = (df['mid_price'] <= 0) | df['mid_price'].isna()
                df.loc[mask_invalid_mid, 'mid_price'] = df.loc[mask_invalid_mid, 'lastPrice']
            
            # Si aún es inválido, usar bid o ask
            mask_still_invalid = (df['mid_price'] <= 0) | df['mid_price'].isna()
            df.loc[mask_still_invalid, 'mid_price'] = df.loc[mask_still_invalid, 'bid']
            mask_still_invalid = (df['mid_price'] <= 0) | df['mid_price'].isna()
            df.loc[mask_still_invalid, 'mid_price'] = df.loc[mask_still_invalid, 'ask']
            
        elif 'lastPrice' in df.columns:
            df['mid_price'] = df['lastPrice']
        else:
            raise ValueError("No se encontraron columnas de precio válidas")
        
        # Eliminar filas donde mid_price sigue siendo inválido
        n_before = len(df)
        df = df[df['mid_price'] > self.min_price].copy()
        print(f"  ✓ Eliminados {n_before - len(df)} registros sin precio medio válido")
        
        # 4. Validar monotonicidad de calls por vencimiento
        print("\n4️⃣ Validando monotonicidad de calls...")
        df = self._validate_call_monotonicity(df)
        
        # 5. Calcular intrinsic value y time value
        print("\n5️⃣ Calculando valores intrínsecos y temporales...")
        df = self._calculate_option_values(df)
        
        # 6. Resumen final
        print(f"\n{'='*60}")
        print("RESUMEN DE LIMPIEZA")
        print(f"{'='*60}")
        print(f"Registros iniciales:    {n_initial}")
        print(f"Registros finales:      {len(df)}")
        print(f"Tasa de retención:      {len(df)/n_initial*100:.1f}%")
        print(f"\nPor tipo:")
        print(df['type'].value_counts())
        print(f"\nPor vencimiento:")
        print(df.groupby('expiration').size().sort_values(ascending=False).head())
        
        return df
    
    def _validate_call_monotonicity(self, df):
        """
        Valida y corrige monotonicidad de precios de calls.
        
        Args:
            df: DataFrame con opciones
            
        Returns:
            DataFrame con validación aplicada
        """
        df_clean = []
        
        for exp_date in df['expiration'].unique():
            for opt_type in ['call', 'put']:
                mask = (df['expiration'] == exp_date) & (df['type'] == opt_type)
                df_subset = df[mask].copy()
                
                if len(df_subset) == 0:
                    continue
                
                # Ordenar por strike
                df_subset = df_subset.sort_values('strike')
                
                if opt_type == 'call':
                    # Para calls: precio debe decrecer con strike
                    # Detectar violaciones
                    prices = df_subset['mid_price'].values
                    strikes = df_subset['strike'].values
                    
                    violations = 0
                    for i in range(1, len(prices)):
                        if prices[i] > prices[i-1]:
                            violations += 1
                    
                    if violations > len(prices) * 0.2:  # Más del 20% violaciones
                        # Aplicar smoothing simple
                        df_subset['mid_price'] = df_subset['mid_price'].rolling(
                            window=3, center=True, min_periods=1
                        ).mean()
                
                df_clean.append(df_subset)
        
        if df_clean:
            df_result = pd.concat(df_clean, ignore_index=True)
            print(f"  ✓ Validación completada para {len(df_result)} opciones")
            return df_result
        else:
            return df
    
    def _calculate_option_values(self, df):
        """
        Calcula valor intrínseco y temporal.
        
        Args:
            df: DataFrame con opciones
            
        Returns:
            DataFrame con valores calculados
        """
        S = df['stockPrice'].iloc[0] if 'stockPrice' in df.columns else None
        
        if S is None:
            print("  ⚠️  Precio del stock no disponible")
            return df
        
        # Valor intrínseco
        df['intrinsic_value'] = 0.0
        
        # Calls
        mask_call = df['type'] == 'call'
        df.loc[mask_call, 'intrinsic_value'] = np.maximum(S - df.loc[mask_call, 'strike'], 0)
        
        # Puts
        mask_put = df['type'] == 'put'
        df.loc[mask_put, 'intrinsic_value'] = np.maximum(df.loc[mask_put, 'strike'] - S, 0)
        
        # Valor temporal
        df['time_value'] = df['mid_price'] - df['intrinsic_value']
        
        # Limpiar valores temporales negativos (pueden indicar arbitraje o datos malos)
        n_negative = (df['time_value'] < -0.01).sum()
        if n_negative > 0:
            print(f"  ⚠️  {n_negative} opciones con valor temporal negativo (posible arbitraje)")
        
        print(f"  ✓ Valores calculados para {len(df)} opciones")
        
        return df
    
    def save_clean_data(self, df, ticker='AAPL', data_dir='data'):
        """
        Guarda datos limpios.
        
        Args:
            df: DataFrame limpio
            ticker: Símbolo del activo
            data_dir: Directorio de salida
        """
        output_dir = Path(data_dir)
        output_file = output_dir / f'{ticker}_options_clean.csv'
        
        df.to_csv(output_file, index=False)
        print(f"\n💾 Datos limpios guardados en: {output_file}")
        
        return output_file


def main():
    """
    Función principal para ejecutar el preprocesamiento.
    """
    from pathlib import Path
    
    # Cargar datos
    data_file = Path('data') / 'AAPL_options.csv'
    
    if not data_file.exists():
        print(f"❌ Error: No se encuentra {data_file}")
        print("   Ejecuta primero data_loader.py")
        return
    
    print(f"📂 Cargando datos desde {data_file}")
    df_options = pd.read_csv(data_file)
    df_options['expiration'] = pd.to_datetime(df_options['expiration'])
    
    # Crear preprocesador
    preprocessor = OptionsPreprocessor(
        min_price=0.01,
        min_volume=0,
        min_oi=0
    )
    
    # Limpiar datos
    df_clean = preprocessor.clean_options_data(df_options)
    
    # Guardar
    preprocessor.save_clean_data(df_clean, ticker='AAPL', data_dir='data')
    
    print("\n✅ Preprocesamiento completado")


if __name__ == "__main__":
    main()