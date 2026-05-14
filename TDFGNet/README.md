## Requirements
- Python 3
- torch
- mmdet
- tqbm
- DCNv2
- scikit-image
<br><br>

## Datasets

./dataset
  DAUB_DTUM
  NUDT-MIRST


## Train
```bash
python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'NUDT-MIRSDT'
python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'IRDST'
python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'DAUB_DTUM'
```
<br>


## Test
```bash

python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'NUDT-MIRSDT' --pth_path ''

python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:0 --epochs 20 --dataset 'IRDST' --pth_path ''

python train.py --model 'TDFGNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:0 --epochs 20 --dataset 'DAUB_DTUM' --pth_path ''

```

