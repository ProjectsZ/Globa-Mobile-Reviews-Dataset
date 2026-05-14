import pandas as pd


def _print_sep(title):
    print(f"\n  --- {title} ---")


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

    _print_sep("Vista General del Dataset")
    n_num = len(df.select_dtypes(include="number").columns)
    n_cat = len(df.select_dtypes(include="object").columns)
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  El dataset contiene {df.shape[0]:,} resenhas y {df.shape[1]} columnas.")
    print(f"  Se identifican {n_num} variables numericas y {n_cat} variables categoricas/texto.")
    print(f"  Memoria total utilizada: {mem:.2f} MB.")
    print(f"  Las marcas cubiertas son: {df[df.columns[df.columns.str.lower().isin(['marca', 'brand'])][0]].unique().tolist()}")
    print(f"  Los paises incluidos son: {df[df.columns[df.columns.str.lower().isin(['pais', 'country'])][0]].unique().tolist()}")
    print(f"  Las fuentes de datos son: {df[df.columns[df.columns.str.lower().isin(['fuente', 'source'])][0]].unique().tolist()}")

    return df


def summary_stats(df: pd.DataFrame) -> None:
    _print_sep("Calidad de los Datos")

    dtypes = df.dtypes
    print("  TIPOS DE DATOS:")
    for col, dtype in dtypes.items():
        print(f"    {col}: {dtype}")

    _print_sep("Valores Nulos")
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df)) * 100
    null_table = pd.DataFrame({"Nulos": nulls, "%": nulls_pct.round(2)})
    nulls_exist = null_table[null_table["Nulos"] > 0]
    if len(nulls_exist) > 0:
        print(nulls_exist)
        print(f"  Se encontraron valores nulos en {len(nulls_exist)} columnas.")
        print(f"  Se aplicara imputacion automatica (numericas con mediana, categoricas con moda).")
    else:
        print("  No se encontraron valores nulos en ninguna columna.")
        print("  El dataset esta completo, no requiere imputacion.")

    _print_sep("Duplicados")
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"  Se detectaron {n_dups} filas duplicadas ({n_dups/len(df)*100:.2f}% del total).")
        print(f"  Seran eliminadas en la etapa de limpieza.")
    else:
        print(f"  No se encontraron filas duplicadas.")

    _print_sep("Consistencia de Variables Categoricas")
    n_checked = 0
    for col in df.select_dtypes(include="object").columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 20:
            n_checked += 1
            print(f"  {col}: {sorted(unique_vals)} ({len(unique_vals)} valores unicos)")
    print(f"  Se revisaron {n_checked} columnas categoricas con baja cardinalidad.")
    print(f"  Estas columnas seran convertidas a tipo 'category' para optimizar memoria.")
