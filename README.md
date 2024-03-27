# HyperTTS : Parameter Efficient Adaptation in Text to Speech using Hypernetworks

Neural speech synthesis, or text-to-speech (TTS), aims to transform a signal from the text domain to the speech domain. While developing TTS architectures that train and test on the same set of speakers has seen significant improvements, out-of-domain speaker performance still faces enormous limitations. Domain adaptation on a new set of speakers can be achieved by fine-tuning the whole model for each new domain, thus making it parameter-inefficient. This problem can be solved by Adapters that provide a parameter-efficient alternative to domain adaptation. Although famous in NLP, speech synthesis has not seen much improvement from Adapters. In this work, we present HyperTTS, which comprises a small learnable network, ``hypernetwork", that generates parameters of the Adapter blocks, allowing us to condition Adapters on speaker representations and making them dynamic. Extensive evaluations of two domain adaptation settings demonstrate its effectiveness in achieving state-of-the-art performance in the parameter-efficient regime. We also compare different variants of HyperTTS, comparing them with baselines in different studies. Promising results on the dynamic adaptation of adapter parameters using hypernetworks open up new avenues for domain-generic multi-speaker TTS systems.

![image](https://github.com/declare-lab/HyperTTS/assets/35062414/6f0ee37b-6f11-4397-a3c9-36bcf3a8042b)



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

## Get objective metrics

```python
python object_metrics.py --ref_wav_dir /data/result/LTS100_GT --synth_wav_dir /data/result/LTS100_syn/
```

## Audio Samples

We compare 20 samples and upload the generated audios to the directory _./Show20Samples_

We refer to this repo:  [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS.git).


