"""Komponen dashboard yang genuinely reusable — nol logic spesifik Prep Presisi (nol
waste_avoided, nol outlet/menu) di sini, supaya bisa diimpor ulang buat dashboard
Streamlit lain. Logic spesifik domain tetap tinggal di _data.py & tiap page."""

from collections.abc import Mapping

import pandas as pd
import streamlit as st


def kpi_row(metrics: Mapping[str, str]) -> None:
    """Render beberapa st.metric dalam satu baris kolom. `metrics` = {label: value_str}."""
    columns = st.columns(len(metrics))
    for col, (label, value) in zip(columns, metrics.items(), strict=True):
        col.metric(label, value)


def data_table(
    df: pd.DataFrame,
    rename: Mapping[str, str],
    number_formats: Mapping[str, str] | None = None,
) -> None:
    """Tabel sortable (klik header) dari kolom-kolom di `rename` (key = kolom asli, value
    = label tampil). Urutan baris ikut `df` yang dikasih — sort/filter dulu di pemanggil
    kalau perlu. `number_formats` opsional: {label_setelah_rename: format string Streamlit
    NumberColumn, mis. "Rp %d"}."""
    display_df = df[list(rename)].rename(columns=rename)
    column_config = {
        col: st.column_config.NumberColumn(format=fmt)
        for col, fmt in (number_formats or {}).items()
    }
    st.dataframe(
        display_df, width="stretch", hide_index=True, column_config=column_config
    )


def entity_picker(
    options: pd.DataFrame, id_col: str, name_col: str, label: str
) -> tuple[str, str]:
    """Selectbox berdasarkan nama, kembalikan (id, name) yang dipilih sebagai string."""
    sorted_options = options.sort_values(name_col)
    selected_name = st.selectbox(label, sorted_options[name_col])
    selected_id = str(
        sorted_options.loc[sorted_options[name_col] == selected_name, id_col].iloc[0]
    )
    return selected_id, selected_name
