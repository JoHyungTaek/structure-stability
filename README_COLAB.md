# structure-stability refactor

이 버전은 `VSCode 수정 -> GitHub push -> Colab에서 git pull -> 학습/추론` 흐름에 맞춰 갈아엎은 구조입니다.

## 핵심 변경점
- `BASE_PATH` 하드코딩 제거
- `argparse + yaml config + override` 구조
- Stage1 `train -> dev` 학습
- Stage2 `dev` 내부 KFold 적응
- temperature scaling 저장
- fold ensemble + TTA inference
- 출력물 전부 `output_root` 아래 저장

## 폴더
```text
structure-stability/
├─ configs/
│  └─ base.yaml
├─ src/
│  ├─ config.py
│  ├─ dataset.py
│  ├─ engine.py
│  ├─ model.py
│  ├─ transforms.py
│  └─ utils.py
├─ train.py
├─ inference.py
├─ requirements.txt
└─ README_COLAB.md
```

## Colab 세팅
```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive
!git clone https://github.com/JoHyungTaek/structure-stability.git
%cd /content/drive/MyDrive/structure-stability
!git pull origin main
!pip install -r requirements.txt
```

## 학습
```python
!python train.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=/content/drive/MyDrive/dacon_structure_stability \
    paths.output_root=/content/drive/MyDrive/structure_outputs \
    train.batch_size=16 \
    model.image_size=320
```

## 추론
```python
!python inference.py \
  --config configs/base.yaml \
  --override \
    paths.data_root=/content/drive/MyDrive/dacon_structure_stability \
    paths.output_root=/content/drive/MyDrive/structure_outputs
```

## VSCode에서 수정 후 반영
```bash
git add .
git commit -m "refactor training pipeline"
git push origin main
```

Colab에서는:
```python
%cd /content/drive/MyDrive/structure-stability
!git pull origin main
```
