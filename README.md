# structure-stability

데이콘 **구조물 안정성 물리 추론 AI 경진대회**용 전체 코드입니다.

이 코드는 다음 대회 조건을 기준으로 작성했습니다.
- 입력은 `front.png`, `top.png` 두 시점 이미지입니다.
- `train`에는 10초짜리 `simulation.mp4`가 포함되지만, 제출 추론에는 사용할 수 없습니다.
- `train.csv`, `dev.csv`에는 `id`, `label`이 있고, `sample_submission.csv`에는 `id`, `unstable_prob`, `stable_prob`가 있습니다. 
- 공개 가중치의 사전학습 모델은 사용할 수 있고, OpenAI/Gemini 같은 원격 API 기반 모델은 사용할 수 없습니다. 

## 폴더 구조

```text
structure-stability/
├─ configs/
│  └─ base.yaml
├─ src/
│  ├─ dataset.py
│  ├─ engine.py
│  ├─ model.py
│  ├─ transforms.py
│  └─ utils.py
├─ train.py
├─ inference.py
├─ requirements.txt
└─ README.md
```

## 모델 개요

- backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- 두 시점 이미지를 **같은 backbone**으로 인코딩
- spatial token으로 펼친 뒤 **cross-view attention fusion** 적용
- `front`, `top`, `abs diff`, `product`, `sum`, `max` feature를 합쳐 binary classifier 수행
- fold별 validation logits로 **temperature scaling** 적용
- inference 시 fold ensemble + hflip TTA 사용

## 데이터 구조

```text
open/
├── train/
│   └── {id}/
│       ├── front.png
│       ├── top.png
│       └── simulation.mp4
├── dev/
│   └── {id}/
│       ├── front.png
│       └── top.png
├── test/
│   └── {id}/
│       ├── front.png
│       └── top.png
├── train.csv
├── dev.csv
└── sample_submission.csv
```

## 로컬 / VSCode 실행

```bash
git clone https://github.com/JoHyungTaek/structure-stability.git
cd structure-stability
pip install -r requirements.txt
```

학습:

```bash
python train.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=./open \
    paths.output_root=./outputs/exp001
```

추론:

```bash
python inference.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=./open \
    paths.output_root=./outputs/exp001
```

## Colab 실행

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf structure-stability
!git clone https://github.com/JoHyungTaek/structure-stability.git
%cd structure-stability
!pip install -r requirements.txt
```

학습:

```python
!python train.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=/content/drive/MyDrive/open \
    paths.output_root=/content/drive/MyDrive/structure_outputs \
    train.batch_size=12 \
    model.image_size=320
```

추론:

```python
!python inference.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=/content/drive/MyDrive/open \
    paths.output_root=/content/drive/MyDrive/structure_outputs
```

## 출력물

- `fold0_best.pt` ... `fold4_best.pt`
- `history.csv`
- `cv_summary.json`
- `submission.csv`

