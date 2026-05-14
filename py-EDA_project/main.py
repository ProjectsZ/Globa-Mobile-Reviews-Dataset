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


def main():
    parser = argparse.ArgumentParser(description="EDA Pipeline - Global Mobile Reviews")
    parser.add_argument("--remove-outliers", action="store_true", help="Eliminar outliers por IQR")
    args = parser.parse_args()

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
    CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "clean_data.csv")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#     PIPELINE EDA - GLOBAL MOBILE REVIEWS DATASET" + " " * 5 + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60 + "\n")

    # --- 1. CARGA ---
    df = load_csv(DATA_PATH)
    summary_stats(df)

    # --- 2. LIMPIEZA ---
    df = adjust_dtypes(df)
    df = handle_missing(df, strategy="auto")
    df = remove_duplicates(df)

    # --- 3. ANALISIS DESCRIPTIVO ---
    stats_df = descriptive_stats(df)

    # --- 4. OUTLIERS ---
    print("\n" + "=" * 60)
    print("DETECCION DE OUTLIERS")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    outlier_info = detect_outliers_iqr(df, cols=numeric_cols)

    for col, info in outlier_info.items():
        if info["count"] > 0:
            print(f"  {col}: {info['count']} outliers ({info['pct']}%) | IQR limites: [{info['lower']:.2f}, {info['upper']:.2f}]")

    if args.remove_outliers:
        df = remove_outliers_iqr(df, cols=numeric_cols)
        print(f"  Dataset despues de eliminar outliers: {df.shape}")

    # --- 5. CORRELACION ---
    corr_matrix = correlation_analysis(df)

    # --- 6. VISUALIZACIONES ---
    generate_all_visualizations(df, corr_matrix)

    # --- 7. EXPORTAR RESULTADOS ---
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n  Dataset limpio exportado a: {CLEAN_PATH}")

    if stats_df is not None:
        stats_df.to_csv(os.path.join(REPORTS_DIR, "descriptive_stats.csv"), index=False)

    if corr_matrix is not None and not corr_matrix.empty:
        corr_matrix.to_csv(os.path.join(REPORTS_DIR, "correlation_matrix.csv"))

    print("\n" + "#" * 60)
    print("#     EDA COMPLETADO EXITOSAMENTE" + " " * 22 + "#")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
