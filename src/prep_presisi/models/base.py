from typing import Protocol

import pandas as pd


class BaseForecaster(Protocol):
    """Kontrak yang wajib diimplementasikan semua model forecasting."""

    def fit(self, train_df: pd.DataFrame) -> None:
        """Melatih model dari DataFrame hasil feature engineering."""
        ...

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        """Mengembalikan DataFrame dengan kolom tambahan `predicted_qty`."""
        ...

    @property
    def name(self) -> str:
        """Identifier model, dipakai untuk logging & perbandingan hasil evaluasi."""
        ...
