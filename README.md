# structure-stability

멀티뷰 이미지(`front.png`, `top.png`) 기반 구조 안정성 예측 코드입니다.

## 이번 수정 방향
- **공식 분리 유지**: `train.csv`, `dev.csv`, `test`만 사용
- **Stage1**: `train -> dev` 사전학습
- **Stage2**: `dev` 내부 `StratifiedKFold` 적응
- **temperature scaling** 적용
- **fold ensemble + TTA inference** 적용
- 제출 컬럼 자동 매핑 보강

## 폴더 구조
```bash
structure-stability/
├─ configs/
│  └─ config.py
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ transforms.py
│  └─ utils.py
├─ train_stagewise.py
├─ refit_dev_all.py
├─ inference.py
└─ requirements.txt
```

## 실행 순서
```bash
python train_stagewise.py
python inference.py
```

refit 실험을 추가로 해보려면:
```bash
python refit_dev_all.py
python inference.py
```

## 주의
- `config.py`에서 `BASE_PATH`를 먼저 맞춰야 합니다.
- 제출 컬럼명이 특이하면 `POSITIVE_LABEL_COL`, `NEGATIVE_LABEL_COL`을 확인하세요.
- 이 코드는 점수 개선용 구조를 반영했지만 특정 점수를 보장하지는 않습니다.
