import pandas as pd
import numpy as np


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("ANALISIS DESCRIPTIVO")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    print(f"\nVariables numericas ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables categoricas ({len(cat_cols)}): {cat_cols}")

    stats = []
    for col in numeric_cols:
        s = df[col]
        stats.append(
            {
                "Variable": col,
                "Tipo": "Numerica",
                "No Nulos": s.count(),
                "Media": round(s.mean(), 2),
                "Mediana": round(s.median(), 2),
                "Moda": s.mode().iloc[0] if not s.mode().empty else np.nan,
                "Varianza": round(s.var(), 2),
                "Desv. Estandar": round(s.std(), 2),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Rango": round(s.max() - s.min(), 2),
                "Asimetria": round(s.skew(), 3),
                "Curtosis": round(s.kurtosis(), 3),
            }
        )

    stats_df = pd.DataFrame(stats)
    print("\n--- Estadisticas Descriptivas (Variables Numericas) ---")
    print(stats_df.to_string(index=False))

    for col in cat_cols:
        if df[col].nunique() < 20:
            print(f"\n--- Distribucion: {col} ---")
            value_counts = df[col].value_counts()
            pcts = df[col].value_counts(normalize=True) * 100
            dist_df = pd.DataFrame({"Frecuencia": value_counts, "%": pcts.round(2)})
            print(dist_df.to_string())

    return stats_df


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CORRELACION DE VARIABLES")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        print("  No hay suficientes variables numericas para correlacion.")
        return pd.DataFrame()

    corr = df[numeric_cols].corr()

    print("\n--- Matriz de Correlacion ---")
    print(corr.to_string())

    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            col_i = corr.columns[i]
            col_j = corr.columns[j]
            if abs(val) > 0.7:
                high_corr.append((col_i, col_j, round(val, 3)))
                print(
                    f"  ALTA CORRELACION: {col_i} <-> {col_j} = {val:.3f}"
                )

    if not high_corr:
        print("  No se encontraron correlaciones fuertes (>0.7).")

    return corr
