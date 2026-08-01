import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Theta

_DEFAULT_NUM_SAMPLE_SERIES = 3
_SEASON_LENGTH = 7  # seasonality mingguan


class StatsForecastForecaster:
    """Model pembanding klasik (time-series decomposition: AutoETS + Theta) — diterapkan
    hanya ke sample series (default 3), bukan semua 850 kombinasi outlet x menu, karena
    StatsForecast secara desain per-series (PRD §6 fase 6, §7). Pure Python/Numba — tidak
    ada risiko compiler C++ seperti Prophet (TRD §3.1).

    `predict_qty` diisi dari AutoETS (default utama); forecast Theta juga disimpan di
    kolom `predicted_qty_theta` untuk observasi kualitatif M4 (PRD §9)."""

    def __init__(
        self,
        series_ids: list[tuple[str, str]] | None = None,
        num_sample_series: int = _DEFAULT_NUM_SAMPLE_SERIES,
    ):
        self._requested_series_ids = series_ids
        self._num_sample_series = num_sample_series
        self._series_ids: list[tuple[str, str]] = []
        self._sf: StatsForecast | None = None
        self._is_fitted = False

    def _select_series(self, train_df: pd.DataFrame) -> list[tuple[str, str]]:
        if self._requested_series_ids is not None:
            return self._requested_series_ids
        # Tanpa series eksplisit, ambil N kombinasi ber-volume tertinggi — paling
        # representatif buat demo pembanding.
        top = (
            train_df.groupby(["outlet_id", "menu_item_id"], observed=True)["qty_sold"]
            .sum()
            .sort_values(ascending=False)
            .head(self._num_sample_series)
        )
        return list(top.index)

    def _to_nixtla_format(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = pd.MultiIndex.from_tuples(self._series_ids)
        mask = df.set_index(["outlet_id", "menu_item_id"]).index.isin(keys)
        subset = df[mask]
        nixtla_df = pd.DataFrame(
            {
                "unique_id": subset["outlet_id"].astype(str)
                + "__"
                + subset["menu_item_id"].astype(str),
                "ds": pd.to_datetime(subset["date"]),
                "y": subset["qty_sold"].astype(float),
            }
        )
        return nixtla_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def fit(self, train_df: pd.DataFrame) -> None:
        self._series_ids = self._select_series(train_df)
        nixtla_df = self._to_nixtla_format(train_df)
        self._sf = StatsForecast(
            models=[
                AutoETS(season_length=_SEASON_LENGTH),
                Theta(season_length=_SEASON_LENGTH),
            ],
            freq="D",
        )
        self._sf.fit(nixtla_df)
        self._is_fitted = True

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "StatsForecastForecaster.predict() dipanggil sebelum fit()"
            )

        h = predict_df["date"].nunique()
        forecast = self._sf.predict(h=h).rename(
            columns={"AutoETS": "predicted_qty", "Theta": "predicted_qty_theta"}
        )
        forecast[["outlet_id", "menu_item_id"]] = forecast["unique_id"].str.split(
            "__", n=1, expand=True
        )
        forecast["date"] = pd.to_datetime(forecast["ds"]).dt.date

        result = predict_df.copy()
        result["outlet_id"] = result["outlet_id"].astype(str)
        result["menu_item_id"] = result["menu_item_id"].astype(str)
        result["date"] = pd.to_datetime(result["date"]).dt.date

        result = result.merge(
            forecast[
                [
                    "outlet_id",
                    "menu_item_id",
                    "date",
                    "predicted_qty",
                    "predicted_qty_theta",
                ]
            ],
            on=["outlet_id", "menu_item_id", "date"],
            how="left",
        )
        return result

    @property
    def name(self) -> str:
        return "statsforecast_sample"
