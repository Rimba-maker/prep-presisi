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
uv run marimo edit notebooks/01_eda_datagen.py   # eksplorasi data (setelah datagen diimplementasikan)
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

## Status implementasi

Struktur project, entities, dan config sudah di-scaffold. Logic di `datagen/`, `features/`, `models/`, `evaluation/`, `insights/`, dan `dashboard/` belum diimplementasikan.
