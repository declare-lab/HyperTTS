## Metrics

These are some of the metrics implemented in this repository as well as tested out from various sources. Their usage has been demonstrated in the examples.py file. Listen to the [clean speech](https://github.com/skit-ai/woc-tts-enhancement/blob/main/docs/audio/sp15.wav) and the [distorted speech](https://github.com/skit-ai/woc-tts-enhancement/blob/main/docs/audio/sp15_station_sn5.wav) and this is how these metrics perform on this audio sample-

```python
>>>import torchaudio
>>>import numpy as np
>>>path1 = "../Testset/clean/sp15.wav" #input the clean audio
>>>path2 = "../Testset/noisy/station/sp15_station_sn5.wav" #input the corresponding noisy audio
>>>y_ref, sr_ref = torchaudio.load(path1) #load the files
>>>y_syn, sr_syn = torchaudio.load(path2)
```

### F0 Frame Error Rate (FFE)

```python
>>>from Metrics.f0_frame_error import FFE
>>>ffe = FFE(22050)
>>>print((ffe.calculate_ffe(y_ref, y_syn))
0.2222222222222222
```

### Gross Pitch Error (GPE)

```python
>>>from Metrics.gross_pitch_error import GPE
>>>gpe = GPE(22050)
>>>print((gpe.calculate_gpe(y_ref, y_syn))
0.0
```

### Mel Cepstral Distortion (MCD)

```python
>>>from Metrics.mcd import MCD
>>>msd = MSD(22050)
>>>print(mcd.calculate_mcd(y_ref, y_syn))
7.930282115936279
```

### MSD

```python
>>>from Metrics.msd import MSD
>>>msd = MSD(22050)
>>>print(msd.calculate_msd(y_ref, y_syn))
3.4365859031677246
```

### Voicing Error Decision (VED)

```python
>>>from Metrics.voicing_decision_error import VDE
>>>vde = VDE(22050)
>>>print(vde(y_ref, y_syn))
0.2222222222222222
```

### Perceptual Evaluation of Speech Quality (PESQ)

```python
>>>from pesq import pesq
>>>print(pesq(sr_ref, y_ref.numpy()[0], y_syn.numpy()[0], "nb"))
1.489499568939209
```

### Short Time Objective Intelligibility (STOI)

```python
>>>from pystoi import stoi
>>>print(stoi(y_ref.numpy()[0], y_syn.numpy()[0], sr_ref, extended=False))
0.7951681427370876
```