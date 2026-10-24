from __future__ import annotations

import numpy as np
import torch

from .data_utils import ExpenseData, build_result

"""
CSV 가계부 데이터
        ↓
ExpenseData로 전처리
 ├─ matrix: TimesFM용 2차원 배열
 └─ long: Chronos용 긴 형식 DataFrame
        ↓
각 모델에서 다음 1개 시점 예측
        ↓
모델별 원시 출력을 공통 형식으로 변환
        ↓
카테고리별 predicted_amount / p10 / p50 / p90
"""

TIMESFM_MODEL_ID = "google/timesfm-3.0-pytorch"
CHRONOS_MODEL_ID = "amazon/chronos-2"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA를 요청했지만 PyTorch가 GPU를 찾지 못했습니다. "
            "--device cpu로 실행하거나 CUDA 설치를 확인하세요."
        )
    return requested


def forecast_timesfm(data: ExpenseData, device: str):
    """Forecast all expense categories jointly with TimesFM 3."""
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(
        checkpoint_path=TIMESFM_MODEL_ID,
        per_core_batch_size=1, # 1개 카테고리씩 예측
        device=device, 
    )
    forecaster = TimesFM3Evaluator(config) # TimesFM3Evaluator 객체 생성

    """
    contexts
    └─ 첫 번째 사용자 데이터
        ├─ 교통 45개월
        ├─ 쇼핑 45개월
        ├─ 식비 45개월
        ├─ 여가 45개월
        └─ 주거 45개월
    """
    output = list(
        forecaster.predict_batch(
            contexts=[data.matrix], # 2차원 배열을 배열로 전달
            horizon=1, # 다음 1개 시점 예측
            return_quantiles=True, # 9개 분위수 반환
            use_symmetric_averaging=False, # 대칭 평균 사용 안 함 (True 시, 원본 시계열과 부호를 반대로 만든 시계열을 둘 다 예측한 뒤 결과를 조합 -> 일부 데이터에서 예측 안정성이 좋아질 수 있음 But 연산 비용이 증가)
        )
    )[0]

    # forecast: (category, horizon), quantiles: (category, horizon, 9 deciles)
    point = np.asarray(output.forecast)[:, 0]
    all_quantiles = np.asarray(output.quantiles)[:, 0, :]
    selected = all_quantiles[:, [0, 4, 8]]
    return build_result(
        model="TimesFM 3",
        categories=data.categories,
        forecast_month=data.next_month,
        point=point,
        quantiles=selected,
    )


def forecast_chronos(data: ExpenseData, device: str):
    """Forecast all expense categories with Chronos-2 DataFrame input."""
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # Chronos용 긴 형식 DataFrame을 모델에 맞게 전처리
    """
    month       category  amount
    2023-01-01  식비       382000
    2023-02-01  식비       361000

    ->
    timestamp   item_id  target
    2023-01-01  식비      382000
    2023-02-01  식비      361000
    """
    context = data.long.rename(
        columns={"category": "item_id", "month": "timestamp", "amount": "target"}
    )
    prediction = pipeline.predict_df(
        context,
        prediction_length=1, # 다음 1개 시점 예측
        quantile_levels=QUANTILE_LEVELS, # 9개 분위수 반환 (필요 분위수만 설정 가능)
        id_column="item_id", # 카테고리 열 이름
        timestamp_column="timestamp", # 날짜 열 이름
        target="target", # 지출액 열 이름
        freq="MS", # 월 단위 예측
        cross_learning=True, # 시계열들을 완전히 따로 예측하지 않고 함께 처리하며 정보를 공유 (즉 식비, 교통, 쇼핑, 여가, 주거 시계열들을 함께 패턴을 참고해 예측)
    )
    prediction = prediction.set_index("item_id").reindex(data.categories)

    return build_result(
        model="Chronos-2",
        categories=data.categories,
        forecast_month=data.next_month,
        point=prediction["predictions"].to_numpy(),
        quantiles=prediction[["0.1", "0.5", "0.9"]].to_numpy(),
    )

