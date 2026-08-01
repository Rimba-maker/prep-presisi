import pandas as pd


class SeasonalNaiveForecaster:
    """Prediksi = qty_sold pada hari yang sama `season_length` hari sebelumnya (default 7
    = minggu lalu) — baseline wajib sebelum model lanjutan dievaluasi (PRD §6 fase 4).
    Butuh kolom `lag_{season_length}` dari features.build_features(), bukan qty_sold
    mentah, supaya predict() bisa dipanggil di val/test tanpa mengintip qty_sold hari itu
    sendiri."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        # Seasonal naive tidak punya parameter untuk dilatih — prediksinya murni membaca
        # kolom lag yang sudah dihitung features.build_features(). fit() di sini cuma
        # menjaga kontrak BaseForecaster (fail loud kalau predict() dipanggil sebelum fit()).
        self._is_fitted = True

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "SeasonalNaiveForecaster.predict() dipanggil sebelum fit()"
            )
        lag_col = f"lag_{self.season_length}"
        if lag_col not in predict_df.columns:
            raise ValueError(
                f"predict_df harus punya kolom '{lag_col}' — jalankan features.build_features() dulu"
            )
        result = predict_df.copy()
        result["predicted_qty"] = result[lag_col]
        return result

    @property
    def name(self) -> str:
        return "seasonal_naive"
