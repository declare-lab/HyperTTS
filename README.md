# HyperTTS

We refer to this repo:  [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS.git).

## Pretrain on LTS
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset LTS
```
## Finetune hyperTTS_all on VCTK or LTS2

```python
# LTS2
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset LTS2 --restore_step 600000
# VCTK
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset VCTK --restore_step 600000
```
## Inference 

```python
CUDA_VISIBLE_DEVICES=2 python3 synthesize.py --source /data/Dataset/preprocessed_data/VCTK_16k/val_unsup.txt --restore_step 900000 --mode batch --dataset VCTK
```

## Get object metrics

```python
python object_metrics.py --ref_wav_dir /data/result/LTS100_GT --synth_wav_dir /data/result/LTS100_syn/
```

## Audio Samples

We compare 20 samples and upload the generated audios to the directory _./Show20Samples_


