# Global Mobile Reviews Dataset — EDA Pipeline

**Global Mobile Reviews Dataset (2025 Edition)** — Research-based, web-scraped global review collection featuring **50,000+ mobile phone reviews** gathered from multiple e-commerce and tech review platforms across **8 countries**.

Covers leading smartphone brands with detailed user opinions, ratings, sentiment polarity, and currency-adjusted pricing data. Each record captures customer experience holistically — spanning demographics, verified purchase details, multi-aspect ratings, and pricing — making this dataset a powerful asset for research, NLP, and analytics.

---

## Project Structure

```
py-EDA_project/
├── data/
│   ├── dataset.csv          # Raw data (50,000 reviews)
│   └── clean_data.csv       # Cleaned data (pipeline output)
├── reports/
│   ├── descriptive_stats.csv         # Descriptive statistics table
│   ├── correlation_matrix.csv        # Pearson correlation matrix
│   └── figures/
│       ├── histogramas.png
│       ├── boxplots.png
│       ├── densidad.png
│       ├── rating_distribution.png
│       ├── price_by_brand.png
│       └── correlation_heatmap.png
├── src/
│   ├── load_data.py         # Data loading & initial inspection
│   ├── cleaning.py          # Data cleaning & type adjustment
│   ├── analysis.py          # Descriptive stats & correlation
│   ├── visualization.py     # All plots (histograms, boxplots, heatmaps, etc.)
│   └── __init__.py
├── main.py                  # Pipeline entry point
├── pyproject.toml           # Project config (uv)
└── README.md
```

---

## Requirements

- Python **>= 3.10**
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

---

## Quick Start

```bash
cd py-EDA_project
uv sync                 # Create virtualenv & install dependencies
uv run python main.py   # Run the full EDA pipeline
```

To also remove outliers detected by IQR:

```bash
uv run python main.py --remove-outliers
```

---

## Dataset Overview

| Attribute | Description |
|---|---|
| **Rows** | 50,000 |
| **Columns** | 25 |
| **Brands** | Apple, Google, Xiaomi, Samsung, OnePlus, Realme, Motorola |
| **Countries** | USA, Brazil, India, UAE, UK, Canada, Australia, Germany |
| **Languages** | English, Portuguese, Hindi, German |
| **Sources** | Amazon, eBay, Flipkart, AliExpress, BestBuy |
| **Time span** | 2022 – 2025 |

### Columns (Spanish)

| Column | Description | Type |
|---|---|---|
| `id_de_resenha` | Review ID | int |
| `nombre_del_cliente` | Customer name | str |
| `edad` | Age | int |
| `marca` | Brand | category |
| `modelo` | Phone model | str |
| `precio_usd` | Price in USD | float |
| `precio_local` | Price in local currency | str |
| `moneda` | Currency code | category |
| `tasa_de_cambio_a_usd` | Exchange rate to USD | float |
| `calificacion` | Rating (1–5) | int |
| `texto_de_resenha` | Review text | str |
| `sentimiento` | Sentiment (Positive/Neutral/Negative) | category |
| `pais` | Country | category |
| `idioma` | Language | category |
| `fecha_de_resenha` | Review date | datetime |
| `compra_verificada` | Verified purchase | bool |
| `calificacion_duracion_bateria` | Battery life rating (1–5) | int |
| `calificacion_camara` | Camera rating (1–5) | int |
| `calificacion_rendimiento` | Performance rating (1–5) | int |
| `calificacion_disenho` | Design rating (1–5) | int |
| `calificacion_pantalla` | Display rating (1–5) | int |
| `longitud_de_resenha` | Review length (characters) | int |
| `numero_de_palabras` | Word count | int |
| `votos_utiles` | Helpful votes | int |
| `fuente` | Source platform | category |

---

## EDA Pipeline — Step by Step

### 1. Data Loading (`src/load_data.py`)

- Reads `dataset.csv` with `pandas.read_csv()`
- Prints **dimensions**, **head()**, **info()**, and **describe()** for an initial look
- Detects null values, duplicates, and unique value counts for categorical columns

**Results on this dataset:**
- 50,000 rows × 25 columns
- **Zero** null values and **zero** duplicates (clean dataset)
- All 7 brands, 8 countries, 4 languages, and 5 sources are well-balanced (~12.5%–20% each)

### 2. Data Cleaning (`src/cleaning.py`)

**2.1 Type adjustment (`adjust_dtypes`)**
- `fecha_de_resenha` → `datetime` (detects columns containing "fecha" or "date")
- `compra_verificada` → `bool`
- Low-cardinality object columns (< 20 unique values) → `category`:
  - `marca` (7), `moneda` (8), `sentimiento` (3), `pais` (8), `idioma` (4), `fuente` (5)
- Excludes free-text columns: `texto_de_resenha`, `nombre_del_cliente`, `precio_local`

**2.2 Missing values (`handle_missing`)**
- Strategy: `"auto"` → numeric columns filled with **median**, categorical with **mode**
- If a column has >50% nulls, the column is dropped
- In this dataset: **no missing values found**

**2.3 Duplicates (`remove_duplicates`)**
- Removes exact duplicate rows
- In this dataset: **0 duplicates found**

### 3. Outlier Detection

Using the **IQR method** (Interquartile Range):
- Lower bound: `Q1 - 1.5 × IQR` | Upper bound: `Q3 + 1.5 × IQR`
- Outliers detected in:
  - `edad`: **190 outliers (0.38%)** — ages outside [3.5, 55.5]
  - `tasa_de_cambio_a_usd`: **6,124 outliers (12.25%)** — INR (83.0) far above other currencies
  - `votos_utiles`: **913 outliers (1.83%)** — votes > 9.5

Optional removal via `--remove-outliers` flag.

### 4. Descriptive Statistics (`src/analysis.py`)

For each numeric variable, computes:
- **Mean, Median, Mode, Variance, Standard Deviation**
- **Min, Max, Range**
- **Skewness** (asymmetry) and **Kurtosis** (tail extremity)

**Key findings:**
| Variable | Mean | Median | Std | Skewness |
|---|---|---|---|---|
| `calificacion` | 3.12 | 3.0 | 1.25 | −0.17 |
| `precio_usd` | $689.69 | $637.04 | $310.31 | 0.56 |
| `edad` | 30.08 | 29.0 | 8.93 | 0.47 |
| `longitud_de_resenha` | 62.79 | 63.0 | 8.02 | 0.25 |
| `votos_utiles` | 3.64 | 3.0 | 2.43 | 0.69 |

Categorical distributions are also printed for variables with <20 categories.

### 5. Correlation Analysis (`src/analysis.py`)

Pearson correlation matrix computed for all numeric variables. Pairs with `|r| > 0.7` are flagged.

**High correlations found:**

| Variable 1 | Variable 2 | r |
|---|---|---|
| `calificacion` | `calificacion_duracion_bateria` | 0.762 |
| `calificacion` | `calificacion_camara` | 0.761 |
| `calificacion` | `calificacion_rendimiento` | 0.756 |
| `calificacion` | `calificacion_disenho` | 0.755 |
| `calificacion` | `calificacion_pantalla` | 0.757 |
| `longitud_de_resenha` | `numero_de_palabras` | 0.780 |

**Interpretation:**
- Overall `calificacion` is strongly correlated with all sub-ratings (battery, camera, performance, design, display) — the overall score reflects real multi-aspect satisfaction.
- `longitud_de_resenha` and `numero_de_palabras` are highly correlated (r=0.78) as expected — they measure the same thing (review length) in different units.
- No correlation between price and rating (r≈0.001), suggesting price does not determine satisfaction.

### 6. Visualizations (`src/visualization.py`)

All figures are saved to `reports/figures/`.

#### Histograms (`histogramas.png`)
Distribution of all 13 numeric variables with 30 bins each.

#### Boxplots (`boxplots.png`)
Boxplots for all numeric variables showing median, IQR, and outliers.

#### Density Plots (`densidad.png`)
Kernel Density Estimation (KDE) plots for numeric distributions.

#### Rating Distribution (`rating_distribution.png`)
- Left: Bar chart of rating frequencies (1–5)
- Right: Bar chart of sentiment counts (Positive/Neutral/Negative)

**Insights:**
- Ratings are roughly centered at 3, slightly left-skewed
- **55% Positive**, 25% Neutral, 20% Negative — a generally positive customer base

#### Price by Brand (`price_by_brand.png`)
Boxplot of `precio_usd` grouped by `marca`, sorted by median price.

**Insights:**
- Apple products have the highest median price
- Realme and Motorola are positioned in the lower price range
- Most brands show significant price spread

#### Correlation Heatmap (`correlation_heatmap.png`)
Triangular heatmap of the Pearson correlation matrix using `RdBu_r` colormap, with values annotated.

### 7. Export Results

All outputs are saved to the `reports/` directory:
| File | Description |
|---|---|
| `reports/descriptive_stats.csv` | Per-variable descriptive statistics |
| `reports/correlation_matrix.csv` | Full Pearson correlation matrix |
| `reports/figures/histogramas.png` | Histogram grid |
| `reports/figures/boxplots.png` | Boxplot grid |
| `reports/figures/densidad.png` | Density plot grid |
| `reports/figures/rating_distribution.png` | Rating & sentiment distribution |
| `reports/figures/price_by_brand.png` | Price distribution per brand |
| `reports/figures/correlation_heatmap.png` | Correlation heatmap |
| `data/clean_data.csv` | Fully cleaned dataset |

---

## Running the Pipeline

```bash
# Crear/actualizar el entorno (Eso instala dependencias del proyecto e ipykernel dentro de .venv.)
uv sync --group dev

# Activar ese entorno en la terminal (PowerShell)
.\.venv\Scripts\Activate.ps1

# Standard run (no outlier removal)
uv run python main.py

# With outlier removal
uv run python main.py --remove-outliers
```

---

## License

This project is for educational and research purposes.

