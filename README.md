# 가계부 시계열 Foundation Model 데모

월별 카테고리 지출을 입력해 다음 달 지출을 다음 두 모델로 예측하는 실험 코드입니다.

- Google TimesFM 3 (`google/timesfm-3.0-pytorch`)
- Amazon Chronos-2 (`amazon/chronos-2`)

모델 성능 검증이 아니라 **입력 형식, 실제 추론 과정, 출력 형태 확인**을 위한 데모입니다.

## 환경

- 프로젝트 내부 Python: 3.12.15 (`.python`)
- 가상환경: `.venv`
- GPU: NVIDIA CUDA 우선, 없으면 CPU 자동 선택

PowerShell에서 가상환경을 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 입력 CSV

기본 예제는 `data/sample_expenses.csv`입니다.

```csv
month,category,amount
2026-07-01,식비,572000
2026-07-01,교통,119000
```

- `month`: 해당 월의 날짜. `YYYY-MM` 또는 `YYYY-MM-DD` 사용 가능
- `category`: 소비 카테고리
- `amount`: 해당 월의 지출액
- 같은 월/카테고리가 여러 행이면 자동 합산
- 빠진 월/카테고리 조합은 지출 0원으로 처리
- 데모 코드가 허용하는 최소 이력은 12개월이며, 실제 실험에는 더 긴 이력을 권장

## 실행

두 모델을 연속 실행합니다. 첫 실행 때 모델 가중치 약 1.8GB가 Hugging Face 캐시에 자동 다운로드됩니다.

```powershell
python run_forecast.py --model both
```

한 모델만 실행할 수도 있습니다.

```powershell
python run_forecast.py --model timesfm
python run_forecast.py --model chronos
```

본인 데이터와 CPU를 명시하려면 다음과 같이 실행합니다.

```powershell
python run_forecast.py --input data\my_expenses.csv --device cpu
```

결과는 `outputs` 아래에 저장됩니다.

- `timesfm_forecast.csv`
- `chronos_forecast.csv`
- `forecast_comparison.csv`
- `forecast_comparison.png`

공통 출력 열은 다음과 같습니다.

- `predicted_amount`: 대표 예측값
- `p10`, `p50`, `p90`: 예측 불확실성을 보여주는 10%, 50%, 90% 분위수
