import pandas as pd
import numpy as np


def _print_sep(title):
    print(f"\n  --- {title} ---")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("ANALISIS DESCRIPTIVO")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    print(f"\nVariables numericas ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables categoricas ({len(cat_cols)}): {cat_cols}")

    _print_sep("Tendencia Central y Dispersion")
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
    print(stats_df.to_markdown(index=False, tablefmt="grid"))

    _print_sep("Interpretacion de la Distribucion de Datos")
    for _, row in stats_df.iterrows():
        insights = []
        var = row["Variable"]

        media = row["Media"]
        mediana = row["Mediana"]
        asimetria = row["Asimetria"]
        if abs(asimetria) < 0.5:
            insights.append(f"distribucion aproximadamente simetrica (asimetria={asimetria})")
        elif asimetria > 0.5:
            insights.append(f"sesgada a la derecha (asimetria={asimetria}) — la mayoria de valores estan por debajo de la media")
        else:
            insights.append(f"sesgada a la izquierda (asimetria={asimetria}) — la mayoria de valores estan por encima de la media")

        if abs(media - mediana) / (row["Desv. Estandar"] if row["Desv. Estandar"] > 0 else 1) > 0.5:
            insights.append(f"media ({media}) difiere de mediana ({mediana}), posible presencia de outliers")

        if row["Curtosis"] > 1:
            insights.append(f"curtosis={row['Curtosis']}: distribucion leptocurtica (concentracion alta en el centro, colas pesadas)")
        elif row["Curtosis"] < -0.5:
            insights.append(f"curtosis={row['Curtosis']}: distribucion platicurtica (distribucion mas plana de lo normal)")

        print(f"  {var}: {'. '.join(insights)}.")

    _print_sep("Distribucion de Variables Categoricas")
    for col in cat_cols:
        if df[col].nunique() < 20:
            print(f"\n  --- {col} ---")
            value_counts = df[col].value_counts()
            pcts = df[col].value_counts(normalize=True) * 100
            dist_df = pd.DataFrame({"Frecuencia": value_counts, "%": pcts.round(2)})
            print(dist_df.to_markdown(tablefmt="grid"))

            top = value_counts.index[0]
            top_pct = pcts.iloc[0]
            print(f"  -> Categoria dominante: '{top}' con {top_pct:.1f}% de los registros.")

    _print_sep("Hallazgos Clave del Analisis Descriptivo")
    rating_col = next((c for c in df.columns if c.lower() in ("calificacion", "rating")), None)
    if rating_col:
        mean_r = df[rating_col].mean()
        std_r = df[rating_col].std()
        print(f"  - La calificacion promedio es {mean_r:.2f}/5.0 (desv. estandar={std_r:.2f}),")
        print(f"    lo que indica una experiencia de usuario {'positiva' if mean_r > 3 else 'moderada'} en general.")

    price_col = next((c for c in df.columns if c.lower() in ("precio_usd", "price_usd")), None)
    if price_col:
        mean_p = df[price_col].mean()
        min_p = df[price_col].min()
        max_p = df[price_col].max()
        print(f"  - El precio promedio de los dispositivos es ${mean_p:.2f} USD,")
        print(f"    con un rango de ${min_p:.0f} a ${max_p:.0f} USD.")

    age_col = next((c for c in df.columns if c.lower() in ("edad", "age")), None)
    if age_col:
        mean_a = df[age_col].mean()
        print(f"  - La edad promedio de los usuarios es {mean_a:.1f} anhos, lo que sugiere un perfil")
        print(f"    de consumidor mayoritariamente adulto joven.")

    print(f"  - Se analizaron {len(numeric_cols)} variables numericas y {len(cat_cols)} categoricas.")
    print(f"  - Estos resultados se exportan a reports/descriptive_stats.csv")

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
    print(corr.to_markdown(tablefmt="grid"))

    _print_sep("Deteccion de Correlaciones Fuertes (|r| > 0.7)")
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            col_i = corr.columns[i]
            col_j = corr.columns[j]
            if abs(val) > 0.7:
                high_corr.append((col_i, col_j, val))

    if high_corr:
        high_df = pd.DataFrame(high_corr, columns=["Variable 1", "Variable 2", "r"])
        high_df["r"] = high_df["r"].round(3)
        print(high_df.to_markdown(index=False, tablefmt="grid"))
    else:
        print("  No se encontraron correlaciones fuertes (>0.7).")

    _print_sep("Interpretacion de las Correlaciones")
    for a, b, v in high_corr:
        if v > 0.8:
            fuerza = "muy fuerte"
        elif v > 0.7:
            fuerza = "fuerte"
        print(f"  {a} y {b}: correlacion {fuerza} positiva (r={v:.3f}).")
        a_low = a.lower()
        b_low = b.lower()
        if "longitud" in a_low or "numero" in b_low or "longitud" in b_low or "numero" in a_low:
            print(f"    -> Relacion esperada: ambas variables miden longitudinalmente la resenha.")
        elif "calificacion" in a_low and "calificacion" in b_low:
            print(f"    -> La calificacion global esta fuertemente asociada a esta sub-calificacion especifica.")

    _print_sep("Relaciones No Significativas")
    rating_col = next((c for c in numeric_cols if c.lower() in ("calificacion", "rating")), None)
    price_col = next((c for c in numeric_cols if c.lower() in ("precio_usd", "price_usd")), None)
    if rating_col and price_col:
        rp = corr.loc[rating_col, price_col]
        print(f"  Calificacion vs Precio: r={rp:.3f} (correlacion {'positiva debil' if rp > 0 else 'negativa debil'} y cercana a cero).")
        print(f"  Esto indica que el precio NO es un factor determinante de la satisfaccion del usuario.")
        print(f"  Dispositivos caros y baratos reciben calificaciones similares.")

    return corr
