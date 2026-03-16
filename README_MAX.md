# structure-stability-max

핵심 변경점
- front/top 동기화 증강
- train simulation.mp4 기반 motion pseudo target 자동 생성
- geometry feature 추가
- dual-view transformer fusion + multi-task head
- StratifiedGroupKFold 적용
- inference TTA 버그 수정

## 학습
```bash
python train.py --config configs/max001.yaml --override paths.data_root=./open paths.output_root=./outputs/max001
```

## 추론
```bash
python inference.py --config configs/max001.yaml --override paths.data_root=./open paths.output_root=./outputs/max001
```

## T4 메모리 부족 시
```bash
python train.py --config configs/max001.yaml --override model.backbone='convnextv2_tiny.fcmae_ft_in22k_in1k' model.hidden_dim=256 train.batch_size=10 train.grad_accum_steps=1 infer.batch_size=16
```
