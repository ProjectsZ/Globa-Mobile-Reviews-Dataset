import argparse
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.load_data import load_csv, summary_stats
from src.cleaning import adjust_dtypes, handle_missing, remove_duplicates, detect_outliers_iqr, remove_outliers_iqr
from src.analysis import descriptive_stats, correlation_analysis
from src.visualization import generate_all_visualizations


ETAPAS = {
    1: {
        "titulo": "ANALISIS DESCRIPTIVO",
        "descripcion": (
            "En esta etapa se realiza un analisis exploratorio inicial del dataset. "
            "Se calculan metricas de tendencia central (media, mediana, moda), "
            "dispersion (varianza, desviacion estandar, rango), y forma de la distribucion "
            "(asimetria y curtosis) para cada variable numerica. Ademas, se examinan las "
            "frecuencias de las variables categoricas para entender la composicion del dataset."
        ),
    },
    2: {
        "titulo": "AJUSTE DE TIPOS DE VARIABLES",
        "descripcion": (
            "Se convierten las columnas a sus tipos de datos mas adecuados para el analisis. "
            "Las fechas se transforman a datetime, las variables booleanas a bool, y las "
            "columnas categoricas con baja cardinalidad (<20 valores unicos) se convierten "
            "a tipo 'category' para optimizar memoria y permitir analisis especificos."
        ),
    },
    3: {
        "titulo": "DETECCION Y TRATAMIENTO DE DATOS AUSENTES",
        "descripcion": (
            "Se identifican valores nulos en el dataset y se aplican estrategias de imputacion. "
            "Las columnas numericas se imputan con la mediana (o media segun la estrategia), "
            "mientras que las categoricas se imputan con la moda. Si una columna supera el 50% "
            "de valores nulos, se elimina del dataset. Tambien se eliminan filas duplicadas."
        ),
    },
    4: {
        "titulo": "IDENTIFICACION DE DATOS ATIPICOS (OUTLIERS)",
        "descripcion": (
            "Se utiliza el metodo del rango intercuartilico (IQR) para detectar valores atipicos "
            "en las variables numericas. Un valor se considera outlier si esta fuera del rango "
            "[Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Se muestra el porcentaje de outliers por variable "
            "y opcionalmente se pueden eliminar con el flag --remove-outliers."
        ),
    },
    5: {
        "titulo": "CORRELACION DE VARIABLES",
        "descripcion": (
            "Se calcula la matriz de correlacion de Pearson entre todas las variables numericas. "
            "Esta matriz revela relaciones lineales entre pares de variables, con valores entre "
            "-1 (correlacion negativa perfecta) y +1 (correlacion positiva perfecta). "
            "Se identifican y reportan las correlaciones fuertes (|r| > 0.7) para su interpretacion."
        ),
    },
}


def mostrar_etapa(numero: int):
    etapa = ETAPAS[numero]
    print("\n" + "#" * 120)
    print(f"# ETAPA {numero}: {etapa['titulo']}")
    print("#" + "-" * 118 + "#")
    for linea in etapa["descripcion"].split(". "):
        print(f"# {linea}.")
    print("#" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EDA Pipeline - Global Mobile Reviews")
    parser.add_argument("--remove-outliers", action="store_true", help="Eliminar outliers por IQR")
    args = parser.parse_args()

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
    CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "clean_data.csv")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "#" * 120)
    print("#" + " " * 118 + "#")
    print("#     PIPELINE EDA - GLOBAL MOBILE REVIEWS DATASET" + " " * 65 + "#")
    print("#" + " " * 118 + "#")
    print("#" * 120 + "\n")

    # --- CARGA INICIAL ---
    print("=" * 60)
    print("CARGA DE DATOS")
    print("=" * 60)
    df = load_csv(DATA_PATH)
    summary_stats(df)

    # --- ETAPA 1: ANALISIS DESCRIPTIVO ---
    mostrar_etapa(1)
    print("  Ejecutando analisis descriptivo sobre las variables numericas y categoricas...\n")
    stats_df = descriptive_stats(df)

    # --- ETAPA 2: AJUSTE DE TIPOS DE VARIABLES ---
    mostrar_etapa(2)
    df = adjust_dtypes(df)

    # --- ETAPA 3: DETECCION Y TRATAMIENTO DE DATOS AUSENTES ---
    mostrar_etapa(3)
    df = handle_missing(df, strategy="auto")
    df = remove_duplicates(df)

    # --- ETAPA 4: IDENTIFICACION DE DATOS ATIPICOS (OUTLIERS) ---
    mostrar_etapa(4)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print("  Aplicando metodo IQR para deteccion de outliers en variables numericas...")
    outlier_info = detect_outliers_iqr(df, cols=numeric_cols)

    outlier_rows = []
    for col, info in outlier_info.items():
        if info["count"] > 0:
            outlier_rows.append({
                "Variable": col,
                "Outliers": info["count"],
                "%": info["pct"],
                "Limite Inferior": round(info["lower"], 2),
                "Limite Superior": round(info["upper"], 2),
            })
    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows)
        print(outlier_df.to_markdown(index=False, tablefmt="grid"))
    else:
        print("  No se detectaron outliers en ninguna variable numerica.")

    if args.remove_outliers:
        df = remove_outliers_iqr(df, cols=numeric_cols)
        print(f"  Dataset despues de eliminar outliers: {df.shape}")

    # --- ETAPA 5: CORRELACION DE VARIABLES ---
    mostrar_etapa(5)
    corr_matrix = correlation_analysis(df)

    # --- VISUALIZACIONES ---
    generate_all_visualizations(df, corr_matrix)

    # --- EXPORTAR RESULTADOS ---
    print("\n" + "=" * 60)
    print("EXPORTACION DE RESULTADOS")
    print("=" * 60)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"  Dataset limpio exportado a: {CLEAN_PATH}")

    if stats_df is not None:
        stats_df.to_csv(os.path.join(REPORTS_DIR, "descriptive_stats.csv"), index=False)
        print(f"  Estadisticas descriptivas exportadas a: {REPORTS_DIR}/descriptive_stats.csv")

    if corr_matrix is not None and not corr_matrix.empty:
        corr_matrix.to_csv(os.path.join(REPORTS_DIR, "correlation_matrix.csv"))
        print(f"  Matriz de correlacion exportada a: {REPORTS_DIR}/correlation_matrix.csv")

    print(f"  Graficos guardados en: {FIGURES_DIR}")

    print("\n" + "#" * 120)
    print("#     EDA COMPLETADO EXITOSAMENTE" + " " * 88 + "#")
    print("#" * 120 + "\n")


if __name__ == "__main__":
    main()
