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
python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'NUDT-MIRSDT'
python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'IRDST'
python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'DAUB_DTUM'
```
<br>


## Test
```bash

python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'NUDT-MIRSDT' --pth_path /home/tcs1/data2/lyc/DTUM_main/results_FAT/HDNet_DTUM_NUDT-MIRSDT_fullySup_align_True/net_20_epoch_1.9730015691145266_loss_2025_10_25__22_31.pth

python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:0 --epochs 20 --dataset 'IRDST' --pth_path /home/tcs1/data2/lyc/DTUM_main/results_FAT/HDNet_DTUM_IRDST_fullySup_align_True/net_20_epoch_35.20146740807427_loss_2025_10_24__20_19.pth

python train.py --model 'HDNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:0 --epochs 20 --dataset 'DAUB_DTUM' --pth_path /home/tcs1/data2/lyc/DTUM_main/results_FAT/HDNet_DTUM_DAUB_DTUM_fullySup_align_True/net_10_epoch_24.89193153283334_loss_2025_12_01__15_49.pth

```

