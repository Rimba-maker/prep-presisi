import pandas as pd
import xgboost as xgb

_FEATURE_COLS = [
    "outlet_id",
    "menu_item_id",
    "day_of_week",
    "is_weekend",
    "is_payday_week",
    "is_ramadan",
    "is_lebaran_week",
    "month",
    "day_of_month",
    "week_of_year",
    "quarter",
    "year",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]
_TARGET_COL = "qty_sold"


class XGBoostForecaster:
    """Global model — satu model dilatih lintas seluruh kombinasi outlet x menu sekaligus,
    dengan outlet_id/menu_item_id sebagai categorical feature native XGBoost
    (`enable_categorical=True`) — bukan one-hot manual (PRD §6 fase 5, TRD §3). Trade-off
    akurasi per-series vs global model diterima sadar (PRD §7)."""

    def __init__(self, val_df: pd.DataFrame | None = None, **xgb_params):
        """`val_df` opsional — kalau diberikan, dipakai untuk early stopping saat fit()
        (TRD §7.1 poin 4). Diinjeksi lewat constructor (bukan parameter fit()) supaya
        fit() tetap match signature BaseForecaster Protocol persis."""
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "enable_categorical": True,
            "tree_method": "hist",
            "early_stopping_rounds": 20,
            "random_state": 42,
        }
        default_params.update(xgb_params)
        self._model = xgb.XGBRegressor(**default_params)
        self._val_df = val_df
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        X_train = train_df[_FEATURE_COLS]
        y_train = train_df[_TARGET_COL]

        fit_kwargs = {}
        if self._val_df is not None:
            fit_kwargs["eval_set"] = [
                (self._val_df[_FEATURE_COLS], self._val_df[_TARGET_COL])
            ]
        else:
            # Tanpa validation set, early stopping tidak bisa jalan (XGBoost error kalau
            # early_stopping_rounds diset tapi eval_set kosong) — matikan.
            self._model.set_params(early_stopping_rounds=None)

        self._model.fit(X_train, y_train, verbose=False, **fit_kwargs)
        self._is_fitted = True

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("XGBoostForecaster.predict() dipanggil sebelum fit()")
        result = predict_df.copy()
        result["predicted_qty"] = self._model.predict(predict_df[_FEATURE_COLS])
        return result

    def feature_importance(self) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("feature_importance() dipanggil sebelum fit()")
        return pd.Series(
            self._model.feature_importances_, index=_FEATURE_COLS
        ).sort_values(ascending=False)

    @property
    def name(self) -> str:
        return "xgboost_global"
