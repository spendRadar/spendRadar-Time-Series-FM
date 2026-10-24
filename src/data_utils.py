from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"month", "category", "amount"}

"""
data.matrix -> TimesFM용 2차원 배열
[
    [교통비 45개월치],
    [쇼핑비 45개월치],
    [식비 45개월치],
    [여가비 45개월치],
    [주거비 45개월치],
]


data.long -> Chronos용 긴 형식 DataFrame
[
    month       category  amount
    2023-01-01  식비       382000
    2023-01-01  교통        75000
    2023-01-01  쇼핑       158000
]
"""

@dataclass(frozen=True)
class ExpenseData:
    """Normalized monthly expense data shared by both model adapters."""

    long: pd.DataFrame
    matrix: np.ndarray
    categories: list[str]
    months: pd.DatetimeIndex

    @property
    def next_month(self) -> pd.Timestamp:
        return self.months[-1] + pd.offsets.MonthBegin(1)


def load_expense_data(csv_path: str | Path) -> ExpenseData:
    """Load, aggregate, and regularize a month/category/amount CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {csv_path}")

    raw = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: " + ", ".join(sorted(missing))
        )

    data = raw.loc[:, ["month", "category", "amount"]].copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise").dt.to_period("M")
    data["category"] = data["category"].astype(str).str.strip()
    data["amount"] = pd.to_numeric(data["amount"], errors="raise")

    if data.empty:
        raise ValueError("CSV에 데이터가 없습니다.")
    if (data["category"] == "").any():
        raise ValueError("빈 카테고리 이름이 있습니다.")
    if (data["amount"] < 0).any():
        raise ValueError("지출액(amount)은 0 이상이어야 합니다.")

    # 같은 달/카테고리가 여러 행이면 거래 내역으로 보고 합산한다.
    grouped = data.groupby(["month", "category"], as_index=False)["amount"].sum()
    months = pd.period_range(
        grouped["month"].min(), grouped["month"].max(), freq="M"
    )
    categories = sorted(grouped["category"].unique().tolist())

    pivot = (
        grouped.pivot(index="month", columns="category", values="amount")
        .reindex(index=months, columns=categories)
        .fillna(0.0)
    )

    if len(pivot) < 12:
        raise ValueError(
            f"월별 관측치가 {len(pivot)}개뿐입니다. 데모에도 최소 12개월을 권장합니다."
        )

    month_index = pivot.index.to_timestamp(how="start")
    normalized_long = (
        pivot.rename_axis(index="month", columns="category")
        .stack(future_stack=True)
        .rename("amount")
        .reset_index()
    )
    normalized_long["month"] = normalized_long["month"].dt.to_timestamp(
        how="start"
    )

    # 모델의 다변량 입력 규약: (카테고리 수, 과거 시점 수)
    matrix = pivot.to_numpy(dtype=np.float32).T
    return ExpenseData(
        long=normalized_long,
        matrix=matrix,
        categories=categories,
        months=month_index,
    )


def build_result(
    *,
    model: str,
    categories: list[str],
    forecast_month: pd.Timestamp,
    point: np.ndarray,
    quantiles: np.ndarray,
) -> pd.DataFrame:
    """Convert model arrays to the common presentation-friendly schema."""
    point = np.asarray(point, dtype=float).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=float)
    if point.shape[0] != len(categories) or quantiles.shape != (
        len(categories),
        3,
    ):
        raise ValueError(
            "예측 결과 크기가 카테고리 수와 맞지 않습니다: "
            f"point={point.shape}, quantiles={quantiles.shape}"
        )

    # 지출액은 음수가 될 수 없으므로 발표용 결과에서는 0으로 잘라낸다.
    return pd.DataFrame(
        {
            "model": model,
            "forecast_month": forecast_month,
            "category": categories,
            "predicted_amount": np.maximum(point, 0.0),
            "p10": np.maximum(quantiles[:, 0], 0.0),
            "p50": np.maximum(quantiles[:, 1], 0.0),
            "p90": np.maximum(quantiles[:, 2], 0.0),
        }
    )

