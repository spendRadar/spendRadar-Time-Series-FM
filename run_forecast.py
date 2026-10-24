from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data_utils import load_expense_data
from src.models import forecast_chronos, forecast_timesfm, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="가계부 월별 지출을 TimesFM 3 / Chronos-2로 예측합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_expenses.csv"),
        help="month,category,amount 열을 가진 CSV 경로",
    )
    parser.add_argument(
        "--model",
        choices=["timesfm", "chronos", "both"],
        default="both",
        help="실행할 모델",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="추론 장치(auto는 CUDA 우선)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="결과 CSV와 그래프를 저장할 폴더",
    )
    return parser.parse_args()


def save_plot(results: pd.DataFrame, output_path: Path) -> None:
    pivot = results.pivot(
        index="category", columns="model", values="predicted_amount"
    )
    ax = pivot.plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set_title("Next-month expense forecast by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted amount (KRW)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data = load_expense_data(args.input)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"입력: {args.input}")
    print(
        f"기간: {data.months[0]:%Y-%m} ~ {data.months[-1]:%Y-%m} "
        f"({len(data.months)}개월)"
    )
    print(f"카테고리: {', '.join(data.categories)}")
    print(f"장치: {device}")
    print("첫 실행에서는 Hugging Face 모델 가중치를 자동으로 다운로드합니다.")

    results: list[pd.DataFrame] = []
    if args.model in {"timesfm", "both"}:
        print("\n[1/2] TimesFM 3 예측 시작")
        timesfm_result = forecast_timesfm(data, device)
        timesfm_result.to_csv(
            args.output_dir / "timesfm_forecast.csv",
            index=False,
            encoding="utf-8-sig",
        )
        results.append(timesfm_result)
        print(timesfm_result.to_string(index=False))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in {"chronos", "both"}:
        print("\n[2/2] Chronos-2 예측 시작")
        chronos_result = forecast_chronos(data, device)
        chronos_result.to_csv(
            args.output_dir / "chronos_forecast.csv",
            index=False,
            encoding="utf-8-sig",
        )
        results.append(chronos_result)
        print(chronos_result.to_string(index=False))

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(
        args.output_dir / "forecast_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_plot(combined, args.output_dir / "forecast_comparison.png")
    print(f"\n결과 저장 완료: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

