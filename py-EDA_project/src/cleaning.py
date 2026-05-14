import pandas as pd
import numpy as np


def _print_sep(title):
    print(f"\n  --- {title} ---")


def adjust_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("AJUSTE DE TIPOS DE VARIABLES")
    print("=" * 60)

    df = df.copy()
    cambios = []

    date_cols = [c for c in df.columns if "date" in c.lower() or "fecha" in c.lower()]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            print(f"  {col} -> datetime")
            cambios.append(col)

    compra_col = next((c for c in df.columns if c.lower() in ("compra_verificada", "verified_purchase")), None)
    if compra_col:
        df[compra_col] = (
            df[compra_col].map({"True": True, "False": False}).astype("bool")
        )
        print(f"  {compra_col} -> bool")
        cambios.append(compra_col)

    cat_cols = (
        df.select_dtypes(include="object")
        .select_dtypes(exclude="datetime")
        .columns.tolist()
    )
    for col in cat_cols:
        n_unique = df[col].nunique()
        low = col.lower()
        if n_unique < 20 and low not in ("texto_de_resenha", "nombre_del_cliente", "precio_local", "review_text", "customer_name", "price_local"):
            df[col] = df[col].astype("category")
            print(f"  {col} -> category ({n_unique} categorias)")
            cambios.append(col)

    _print_sep("Resumen de Conversion de Tipos")
    print(f"  Se ajustaron {len(cambios)} columnas:")
    for c in cambios:
        print(f"    - {c}: ahora es {df[c].dtype}")
    print(f"  Con estos ajustes, el uso de memoria se reduce y las variables quedan")
    print(f"  en el formato correcto para el analisis numerico y grafico.")

    return df


def handle_missing(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("TRATAMIENTO DE DATOS AUSENTES")
    print("=" * 60)

    df = df.copy()
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]

    if len(null_cols) == 0:
        print("  No hay valores nulos en el dataset.")
        return df

    _print_sep("Deteccion de Valores Nulos")
    print(f"  Se encontraron {len(null_cols)} columnas con valores ausentes.")
    print(f"  Estrategia de imputacion: {strategy}")

    for col in null_cols.index:
        n_nulls = nulls[col]
        pct = (n_nulls / len(df)) * 100
        print(f"  {col}: {n_nulls} nulos ({pct:.2f}%)")

        if pct > 50:
            print(f"    -> Eliminando columna (>{50}% nulos)")
            df.drop(columns=[col], inplace=True)
            continue

        if np.issubdtype(df[col].dtype, np.number):
            if strategy == "auto":
                fill_val = df[col].median()
                method_name = "mediana"
            elif strategy == "mean":
                fill_val = df[col].mean()
                method_name = "media"
            elif strategy == "median":
                fill_val = df[col].median()
                method_name = "mediana"
            elif strategy == "mode":
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                method_name = "moda"

            df[col].fillna(fill_val, inplace=True)
            print(f"    -> Imputados con {method_name} ({fill_val:.2f})")
        else:
            fill_val = (
                df[col].mode().iloc[0]
                if not df[col].mode().empty
                else "Unknown"
            )
            df[col].fillna(fill_val, inplace=True)
            print(f"    -> Imputados con moda ('{fill_val}')")

    _print_sep("Resultado de Imputacion")
    remaining = df.isnull().sum().sum()
    print(f"  Valores nulos restantes: {remaining}")
    print(f"  El dataset ahora esta completamente limpio y listo para analisis.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"\n  Duplicados eliminados: {removed}")
    if removed > 0:
        print(f"  El dataset se redujo de {before:,} a {after:,} filas ({removed/before*100:.2f}%).")

    _print_sep("Resumen de Limpieza")
    print(f"  Dataset final: {after:,} filas x {df.shape[1]} columnas")
    print(f"  El proceso de limpieza ha finalizado. Los datos estan listos para")
    print(f"  el analisis descriptivo y modelado.")

    return df


def detect_outliers_iqr(df: pd.DataFrame, cols: list = None) -> dict:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    outlier_info = {}

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = {
            "count": len(outliers),
            "pct": round(len(outliers) / len(df) * 100, 2),
            "lower": lower,
            "upper": upper,
        }

    return outlier_info


def detect_outliers_zscore(
    df: pd.DataFrame, cols: list = None, threshold: float = 3.0
) -> dict:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    outlier_info = {}
    for col in cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > threshold]
        outlier_info[col] = {
            "count": len(outliers),
            "pct": round(len(outliers) / len(df) * 100, 2),
        }

    return outlier_info


def remove_outliers_iqr(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    total_before = len(df_clean)

    if cols is None:
        cols = df_clean.select_dtypes(include=np.number).columns.tolist()

    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    removed = total_before - len(df_clean)
    print(
        f"\n  Outliers eliminados (IQR): {removed} filas ({removed/total_before*100:.2f}%)"
    )
    return df_clean
