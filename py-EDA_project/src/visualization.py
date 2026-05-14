import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


FIGS_DIR = "reports/figures"
B = "\033[1m"
R = "\033[0m"


def set_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_histograms(df: pd.DataFrame, grafico_num: int, cols: list = None) -> int:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    n_cols = min(len(cols), 12)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(cols[:n_cols]):
        ax = axes[i]
        df[col].hist(bins=30, ax=ax, edgecolor="white", color="steelblue")
        ax.set_title(f"Histograma: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/histogramas.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Histogramas{R} — distribucion de frecuencias de variables numericas")
    for col in cols[:n_cols]:
        s = df[col].dropna()
        skew = s.skew()
        if abs(skew) < 0.5:
            forma = "aproximadamente simetrica"
        elif skew > 0.5:
            forma = "sesgada a la derecha (asimetria positiva)"
        else:
            forma = "sesgada a la izquierda (asimetria negativa)"
        print(f"  {col}: media={s.mean():.2f}, mediana={s.median():.2f}, asimetria={skew:.3f} -> distribucion {forma}")
    print(f"  Archivo: histogramas.png")
    return grafico_num + 1


def plot_boxplots(df: pd.DataFrame, grafico_num: int, cols: list = None) -> int:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    n_cols = min(len(cols), 12)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(cols[:n_cols]):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax, color="steelblue")
        ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/boxplots.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Boxplots{R} — diagramas de caja y bigotes para deteccion de outliers")
    for col in cols[:n_cols]:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        print(f"  {col}: mediana={s.median():.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, outliers={len(outliers)} ({len(outliers)/len(s)*100:.1f}%)")
    print(f"  Archivo: boxplots.png")
    return grafico_num + 1


def plot_density(df: pd.DataFrame, grafico_num: int, cols: list = None) -> int:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    n_cols = min(len(cols), 6)
    n_rows = (n_cols + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(cols[:n_cols]):
        ax = axes[i]
        df[col].plot.kde(ax=ax, color="steelblue", linewidth=2)
        ax.set_title(f"Distribucion (Densidad): {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidad")
        ax.fill_between([], [], alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/densidad.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Graficos de Densidad{R} — estimacion de densidad kernel (KDE)")
    for col in cols[:n_cols]:
        s = df[col].dropna()
        kurt = s.kurtosis()
        if kurt > 0:
            cola = "cola pesada (leptocurtica, mas outliers de lo normal)"
        elif kurt < 0:
            cola = "cola ligera (platicurtica, menos outliers de lo normal)"
        else:
            cola = "cola similar a la normal (mesocurtica)"
        print(f"  {col}: curtosis={kurt:.3f} -> {cola}")
    print(f"  Archivo: densidad.png")
    return grafico_num + 1


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, grafico_num: int, title: str = "Matriz de Correlacion") -> int:
    if corr_matrix.empty:
        return grafico_num

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Heatmap de Correlacion{R} — matriz de correlacion de Pearson entre variables numericas")
    print(f"  Valores cercanos a +1 (rojo) indican correlacion positiva fuerte.")
    print(f"  Valores cercanos a -1 (azul) indican correlacion negativa fuerte.")
    print(f"  Valores cercanos a 0 (blanco) indican correlacion debil o nula.")
    high_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                high_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
    if high_pairs:
        print(f"  Correlaciones altas (>0.7) encontradas:")
        for a, b, v in high_pairs:
            print(f"    {a} <-> {b}: r={v:.3f}")
    else:
        print(f"  No se encontraron correlaciones altas (>0.7).")
    print(f"  Archivo: correlation_heatmap.png")
    return grafico_num + 1


def plot_rating_distribution(df: pd.DataFrame, grafico_num: int) -> int:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    col_map = {c.lower(): c for c in df.columns}
    rating_col = col_map.get("calificacion", col_map.get("rating", "rating"))
    sentiment_col = col_map.get("sentimiento", col_map.get("sentiment", "sentiment"))

    rating_counts = df[rating_col].value_counts().sort_index()
    axes[0].bar(rating_counts.index, rating_counts.values, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribucion de Ratings", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_xticks(rating_counts.index)

    sentiment_counts = df[sentiment_col].value_counts()
    colors = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
    bars = axes[1].bar(
        sentiment_counts.index,
        sentiment_counts.values,
        color=[colors.get(s, "steelblue") for s in sentiment_counts.index],
        edgecolor="white",
    )
    axes[1].set_title("Distribucion de Sentimiento", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sentimiento")
    axes[1].set_ylabel("Frecuencia")

    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(int(bar.get_height())),
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/rating_distribution.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Distribucion de Ratings y Sentimiento{R} — frecuencias de calificaciones (1-5) y polaridad de sentimiento")
    total = len(df)
    rating_table = pd.DataFrame({
        "Rating": rating_counts.index,
        "Frecuencia": rating_counts.values,
        "%": [f"{c/total*100:.1f}" for c in rating_counts.values]
    })
    print(f"  Rating promedio: {df[rating_col].mean():.2f} / 5.0 (mediana: {df[rating_col].median():.0f})")
    print(f"  Distribucion de calificaciones:")
    print(rating_table.to_markdown(index=False, tablefmt="grid"))
    sent_table = pd.DataFrame({
        "Sentimiento": sentiment_counts.index,
        "Frecuencia": sentiment_counts.values,
        "%": [f"{c/total*100:.1f}" for c in sentiment_counts.values]
    })
    print(f"  Distribucion de sentimiento:")
    print(sent_table.to_markdown(index=False, tablefmt="grid"))
    positivo = sentiment_counts.get("Positive", 0)
    print(f"  La mayoria de las resenhas son {('positivas' if positivo > total/2 else 'negativas o neutrales')} ({positivo/total*100:.1f}% Positive).")
    print(f"  Archivo: rating_distribution.png")
    return grafico_num + 1


def plot_price_by_brand(df: pd.DataFrame, grafico_num: int) -> int:
    col_map = {c.lower(): c for c in df.columns}
    brand_col = col_map.get("marca", col_map.get("brand", "brand"))
    price_col = col_map.get("precio_usd", col_map.get("price_usd", "price_usd"))
    if brand_col not in df.columns or price_col not in df.columns:
        return grafico_num

    fig, ax = plt.subplots(figsize=(14, 6))
    brand_order = df.groupby(brand_col)[price_col].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x=brand_col, y=price_col, order=brand_order, ax=ax, palette="Set2", hue=brand_col, legend=False)
    ax.set_title("Distribucion de Precio por Marca", fontsize=13, fontweight="bold")
    ax.set_xlabel("Marca")
    ax.set_ylabel("Precio (USD)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{FIGS_DIR}/price_by_brand.png", bbox_inches="tight")
    plt.close()

    print(f"\n  {B}Grafico {grafico_num}: Precio por Marca{R} — distribucion de precios USD agrupados por marca")
    stats = df.groupby(brand_col)[price_col].agg(["mean", "median", "min", "max", "std"]).sort_values("median", ascending=False)
    stats = stats.round(0).astype(int)
    stats.columns = ["Media", "Mediana", "Min", "Max", "Std"]
    print(stats.to_markdown(tablefmt="grid"))
    mas_cara = stats.index[0]
    mas_barata = stats.index[-1]
    print(f"  La marca mas cara es {mas_cara} (mediana=${stats.iloc[0]['Mediana']:.0f}).")
    print(f"  La marca mas economica es {mas_barata} (mediana=${stats.iloc[-1]['Mediana']:.0f}).")
    diff = stats.iloc[0]["Mediana"] - stats.iloc[-1]["Mediana"]
    print(f"  Diferencia de mediana entre la mas cara y la mas barata: ${diff:.0f}.")
    print(f"  Archivo: price_by_brand.png")
    return grafico_num + 1


def generate_all_visualizations(df: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> None:
    import os
    os.makedirs(FIGS_DIR, exist_ok=True)

    set_style()

    print("\n" + "=" * 60)
    print("GENERACION DE VISUALIZACIONES")
    print("=" * 60)

    n = 1
    n = plot_histograms(df, n)
    n = plot_boxplots(df, n)
    n = plot_density(df, n)
    n = plot_rating_distribution(df, n)
    n = plot_price_by_brand(df, n)

    if corr_matrix is not None and not corr_matrix.empty:
        n = plot_correlation_heatmap(corr_matrix, n)
