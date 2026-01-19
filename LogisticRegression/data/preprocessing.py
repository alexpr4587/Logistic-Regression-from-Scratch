import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def normalize_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # Solo normalizamos las características (X), no la etiqueta (y)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    features_to_normalize = [col for col in numeric_cols if col != target_column]
    
    # Aplicamos Z-score normalization: (x - media) / desviación estándar
    df[features_to_normalize] = (df[features_to_normalize] - df[features_to_normalize].mean()) / df[features_to_normalize].std()
    return df

def split_features_target(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def preprocess_data(file_path, target_column):
    df = load_data(file_path)
    cleaned_df = clean_data(df)
    
    # ¡ESTA ES LA LÍNEA QUE DEBES CORREGIR!
    # Ahora pasamos target_column para que no normalice la 'y'
    normalized_df = normalize_data(cleaned_df, target_column) 
    
    X, y = split_features_target(normalized_df, target_column)
    processed_df = normalized_df.copy()
    processed_df[target_column] = y
    return processed_df
