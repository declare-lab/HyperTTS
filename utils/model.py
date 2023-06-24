import os
import json

import torch
import numpy as np

import hifigan
from model import CompTransTTS, ScheduledOptim

from transformers import get_constant_schedule_with_warmup

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = CompTransTTS(preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    if train:
        # scheduled_optim = ScheduledOptim(
        #     model, train_config, model_config, args.restore_step
        # )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        soft_optimizer_parameters = [{'params': [p for name, p in model.named_parameters() if ('hypernet' in name) ]}]
        optimizer =  torch.optim.Adam(
            soft_optimizer_parameters,
            lr= 1e-4,  ## added
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        scheduled_optim = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
        model.train()
        return model, optimizer, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "speechbrain-hifigan":
        from speechbrain.pretrained import HIFIGAN
        vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir")

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "speechbrain-hifigan":
            wavs = vocoder.decode_batch(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
