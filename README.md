# Prep Presisi

Demand forecasting practice project — forecast qty prep per outlet × menu item × hari untuk mengurangi waste bahan baku, di skenario chain rumah makan multi-outlet.

## Setup

```bash
uv sync
cp .env.example .env   # opsional — isi OPENROUTER_API_KEY kalau mau coba fitur insight naratif
```

`.env` cuma dibutuhkan untuk fitur LLM insight di dashboard (fitur opsional). Seluruh pipeline inti (datagen → training → dashboard) jalan tanpa itu.

## Menjalankan tiap tahap

```bash
uv run python -m prep_presisi.datagen              # generate data sintetis ke data/raw/
uv run marimo edit notebooks/01_eda_datagen.py      # EDA
uv run python -m prep_presisi.models seasonal_naive     # baseline
uv run python -m prep_presisi.models xgboost_global      # model utama
uv run python -m prep_presisi.models statsforecast_sample  # model pembanding (stretch)
uv run pytest                                     # testing
uv run ruff check . --fix                         # lint
uv run ruff format .                              # format
uv run streamlit run src/prep_presisi/dashboard/app.py   # dashboard (setelah model dilatih)
```

## Struktur

- `config/` — business rules & parameter simulasi (TOML, bukan hardcoded)
- `src/prep_presisi/entities/` — domain model Pydantic
- `src/prep_presisi/datagen/` — generator data sintetis
- `src/prep_presisi/features/` — feature engineering + train/val/test split
- `src/prep_presisi/models/` — forecaster (Protocol `BaseForecaster`, Open/Closed — model baru = file baru)
- `src/prep_presisi/evaluation/` — MAPE/WMAPE + estimasi waste avoided
- `src/prep_presisi/insights/` — insight naratif LLM (LangChain + OpenRouter, opsional)
- `src/prep_presisi/dashboard/` — Streamlit app

## Hasil Eksperimen (test set: 2025-11-01 s/d 2025-12-31)

| Model | MAPE | WMAPE | Catatan |
|---|---|---|---|
| Seasonal Naive (baseline) | 12.06% | 11.88% | Reference point wajib (M2) |
| XGBoost Global | **8.01%** | **7.74%** | Mengalahkan baseline +33.6%/+34.8% (M3). Feature importance: `rolling_mean_7` + `lag_1` mendominasi (~85%), `is_lebaran_week` sinyal event terkuat. |
| StatsForecast (AutoETS/Theta, 3 sample series) | 7.74% | 7.52% | (M4) |

**Observasi kualitatif M4:** dibandingkan apple-to-apple di 3 series volume tertinggi yang sama, XGBoost tetap sedikit lebih unggul (MAPE 7.30% vs StatsForecast 7.74%) — model global dengan lag/rolling features & cross-learning ternyata cukup kompetitif bahkan di level series individual, bukan cuma unggul di rata-rata lintas seluruh kombinasi. Trade-off global-vs-per-series (PRD §7) di project ini tidak terlalu terasa merugikan akurasi individual.

Log lengkap tiap training run: `artifacts/experiments.jsonl` (di-gitignore, regenerated).

## Status implementasi

Semua fase (1-9) sudah diimplementasikan dan tervalidasi end-to-end: data generator, EDA, feature engineering, baseline, XGBoost, StatsForecast, evaluation (metric + waste avoided Rupiah), dashboard Streamlit (2 halaman, teruji via browser), dan insight naratif LLM (LangChain + OpenRouter, opsional, graceful fallback teruji). Definition of Done (PRD §9.1) terpenuhi.
