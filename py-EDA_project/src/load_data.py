import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATOS SIN PROCESAR")
    print("=" * 60)
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    print("--- head() ---")
    print(df.head().to_string(index=False))
    print("\n--- info() ---")
    df.info()
    print("\n--- describe() ---")
    print(df.describe(include="all").to_string())
    return df


def summary_stats(df: pd.DataFrame) -> None:
    print("\n--- TIPOS DE DATOS ---")
    print(df.dtypes)
    print("\n--- VALORES NULOS ---")
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df)) * 100
    null_table = pd.DataFrame({"Nulos": nulls, "%": nulls_pct.round(2)})
    print(null_table[null_table["Nulos"] > 0])
    print("\n--- DUPLICADOS ---")
    print(f"Filas duplicadas: {df.duplicated().sum()}")
    print("\n--- INCONSISTENCIAS ---")
    for col in df.select_dtypes(include="object").columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 20:
            print(f"{col}: {unique_vals}")
