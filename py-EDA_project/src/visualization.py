import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


FIGS_DIR = "reports/figures"


def set_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_histograms(df: pd.DataFrame, cols: list = None) -> None:
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

    print(f"\n  --- Interpretacion: Histogramas ---")
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
    print(f"  Histogramas guardados en {FIGS_DIR}/histogramas.png")


def plot_boxplots(df: pd.DataFrame, cols: list = None) -> None:
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

    print(f"\n  --- Interpretacion: Boxplots ---")
    for col in cols[:n_cols]:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        print(f"  {col}: mediana={s.median():.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, outliers={len(outliers)} ({len(outliers)/len(s)*100:.1f}%)")
    print(f"  Boxplots guardados en {FIGS_DIR}/boxplots.png")


def plot_density(df: pd.DataFrame, cols: list = None) -> None:
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

    print(f"\n  --- Interpretacion: Graficos de Densidad ---")
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
    print(f"  Graficos de densidad guardados en {FIGS_DIR}/densidad.png")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Matriz de Correlacion") -> None:
    if corr_matrix.empty:
        return

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

    print(f"\n  --- Interpretacion: Heatmap de Correlacion ---")
    print(f"  El heatmap muestra la matriz de correlacion de Pearson entre variables numericas.")
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
    print(f"  Heatmap de correlacion guardado en {FIGS_DIR}/correlation_heatmap.png")


def plot_rating_distribution(df: pd.DataFrame) -> None:
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

    print(f"\n  --- Interpretacion: Distribucion de Ratings y Sentimiento ---")
    total = len(df)
    print(f"  Rating promedio: {df[rating_col].mean():.2f} / 5.0 (mediana: {df[rating_col].median():.0f})")
    print(f"  Distribucion de calificaciones:")
    for r, c in rating_counts.items():
        print(f"    Rating {r}: {c} ({c/total*100:.1f}%)")
    print(f"  Distribucion de sentimiento:")
    for s, c in sentiment_counts.items():
        print(f"    {s}: {c} ({c/total*100:.1f}%)")
    positivo = sentiment_counts.get("Positive", 0)
    print(f"  La mayoria de las resenhas son {('positivas' if positivo > total/2 else 'negativas o neutrales')} ({positivo/total*100:.1f}% Positive).")
    print(f"  Distribucion de ratings guardada en {FIGS_DIR}/rating_distribution.png")


def plot_price_by_brand(df: pd.DataFrame) -> None:
    col_map = {c.lower(): c for c in df.columns}
    brand_col = col_map.get("marca", col_map.get("brand", "brand"))
    price_col = col_map.get("precio_usd", col_map.get("price_usd", "price_usd"))
    if brand_col not in df.columns or price_col not in df.columns:
        return

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

    print(f"\n  --- Interpretacion: Precio por Marca ---")
    stats = df.groupby(brand_col)[price_col].agg(["mean", "median", "min", "max", "std"]).sort_values("median", ascending=False)
    for marca, row in stats.iterrows():
        print(f"  {marca}: media=${row['mean']:.0f}, mediana=${row['median']:.0f}, rango=[${row['min']:.0f}-${row['max']:.0f}]")
    mas_cara = stats.index[0]
    mas_barata = stats.index[-1]
    print(f"  La marca mas cara es {mas_cara} (mediana=${stats.iloc[0]['median']:.0f}).")
    print(f"  La marca mas economica es {mas_barata} (mediana=${stats.iloc[-1]['median']:.0f}).")
    diff = stats.iloc[0]["median"] - stats.iloc[-1]["median"]
    print(f"  Diferencia de mediana entre la mas cara y la mas barata: ${diff:.0f}.")
    print(f"  Precio por marca guardado en {FIGS_DIR}/price_by_brand.png")


def generate_all_visualizations(df: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> None:
    import os
    os.makedirs(FIGS_DIR, exist_ok=True)

    set_style()

    print("\n" + "=" * 60)
    print("GENERACION DE VISUALIZACIONES")
    print("=" * 60)

    plot_histograms(df)
    plot_boxplots(df)
    plot_density(df)
    plot_rating_distribution(df)
    plot_price_by_brand(df)

    if corr_matrix is not None and not corr_matrix.empty:
        plot_correlation_heatmap(corr_matrix)
