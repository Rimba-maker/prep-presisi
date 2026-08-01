import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    return Path, alt, mo, pd


@app.cell
def _(mo):
    mo.md(
        "# EDA — Data Generator Prep Presisi\n\n"
        "Validasi bahwa data sintetis di `data/raw/` masuk akal sebelum dipakai untuk "
        "feature engineering & modeling (PRD §6 fase 2)."
    )


@app.cell
def _(Path, pd):
    RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    outlets = pd.read_parquet(RAW_DIR / "outlets.parquet")
    menu_items = pd.read_parquet(RAW_DIR / "menu_items.parquet")
    sales = pd.read_parquet(RAW_DIR / "sales_records.parquet")
    return menu_items, outlets, sales


@app.cell
def _(mo, sales):
    n_negative_prepared = (sales.qty_prepared < 0).sum()
    n_negative_sold = (sales.qty_sold < 0).sum()
    n_sold_exceeds_prepared = (sales.qty_sold > sales.qty_prepared).sum()

    validation_summary = mo.md(f"""
    ## Validasi Dasar

    - Total baris: **{len(sales):,}**
    - Baris `qty_prepared` negatif: **{n_negative_prepared}**
    - Baris `qty_sold` negatif: **{n_negative_sold}**
    - Baris `qty_sold` > `qty_prepared` (harus 0, ini invariant): **{n_sold_exceeds_prepared}**
    """)
    validation_summary


@app.cell
def _(mo, outlets):
    mo.md(f"## Outlets & Menu Items\n\n{len(outlets)} outlet ter-generate.")


@app.cell
def _(outlets):
    outlets


@app.cell
def _(menu_items):
    menu_items


@app.cell
def _(mo):
    mo.md("## Seasonality — Time Series Harian")


@app.cell
def _(sales):
    daily_totals = sales.groupby("date", as_index=False).qty_sold.sum()
    return (daily_totals,)


@app.cell
def _(alt, daily_totals):
    seasonality_chart = (
        alt.Chart(daily_totals)
        .mark_line()
        .encode(x="date:T", y="qty_sold:Q")
        .properties(
            title="Total qty_sold harian (2 tahun) — cari pola mingguan & lonjakan/dip Ramadan-Lebaran",
            width=700,
            height=300,
        )
    )
    seasonality_chart


@app.cell
def _(mo):
    mo.md("## Efek Business Rules (weekend, payday, Ramadan, Lebaran)")


@app.cell
def _(pd, sales):
    effect_comparison = pd.concat(
        [
            sales.groupby("is_weekend")
            .qty_sold.mean()
            .rename("avg_qty_sold")
            .reset_index()
            .rename(columns={"is_weekend": "active"})
            .assign(effect="weekend"),
            sales.groupby("is_payday_week")
            .qty_sold.mean()
            .rename("avg_qty_sold")
            .reset_index()
            .rename(columns={"is_payday_week": "active"})
            .assign(effect="payday"),
            sales.groupby("is_ramadan")
            .qty_sold.mean()
            .rename("avg_qty_sold")
            .reset_index()
            .rename(columns={"is_ramadan": "active"})
            .assign(effect="ramadan"),
            sales.groupby("is_lebaran_week")
            .qty_sold.mean()
            .rename("avg_qty_sold")
            .reset_index()
            .rename(columns={"is_lebaran_week": "active"})
            .assign(effect="lebaran_week"),
        ],
        ignore_index=True,
    )
    return (effect_comparison,)


@app.cell
def _(alt, effect_comparison):
    effect_chart = (
        alt.Chart(effect_comparison)
        .mark_bar()
        .encode(
            x=alt.X("effect:N", title="Business rule"),
            xOffset="active:N",
            y=alt.Y("avg_qty_sold:Q", title="Rata-rata qty_sold"),
            color=alt.Color("active:N", title="Aktif?"),
        )
        .properties(
            title="Efek tiap business rule terhadap rata-rata qty_sold",
            width=500,
            height=300,
        )
    )
    effect_chart


@app.cell
def _(mo):
    mo.md("## Distribusi Antar Outlet")


@app.cell
def _(sales):
    outlet_avg = sales.groupby("outlet_id", as_index=False).qty_sold.mean()
    return (outlet_avg,)


@app.cell
def _(alt, outlet_avg):
    outlet_dist_chart = (
        alt.Chart(outlet_avg)
        .mark_bar()
        .encode(
            x=alt.X(
                "qty_sold:Q",
                bin=alt.Bin(maxbins=30),
                title="Rata-rata qty_sold per outlet",
            ),
            y=alt.Y("count():Q", title="Jumlah outlet"),
        )
        .properties(
            title="Distribusi rata-rata qty_sold antar outlet (harus bervariasi, bukan seragam)",
            width=500,
            height=300,
        )
    )
    outlet_dist_chart


@app.cell
def _(mo, sales):
    waste_pct = (sales.qty_prepared - sales.qty_sold).sum() / sales.qty_prepared.sum()
    mo.md(
        f"## Waste\n\nTotal waste sebagai persentase total qty_prepared: **{waste_pct:.1%}**"
    )


if __name__ == "__main__":
    app.run()
