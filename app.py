#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import math
import torch
import soundfile as sf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.io.wavfile import write

# VITS
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

from flask import Flask, request, send_file
app = Flask(__name__)
hps = utils.get_hparams_from_file("./configs/biaobei_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()
_ = utils.load_checkpoint("/root/VITS/VITS-Paimon_930.pth", net_g, None)
print("Load checkpoint success!")


# 参数说明
#  text: 转换的文本
#  scale: 语音长度，默认1.0
@app.get("/vits")
def vits_transforms():
    text = request.args.get('text', None)
    scale = float(request.args.get('scale', '1.0'))
    print("[length_scale={}] {}".format(scale, text.encode('utf-8')))

    output = f'results/test.wav'
    transform_wave(text, output=output, length_scale=scale)
    return send_file(output, mimetype="audio/wav")
    #  as_attachment=True, attachment_filename="transforms.wav")


def transform_wave(text, output=f'results/test.wav', length_scale=1.2, emotion=0.667, morpheme=0.8):
    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=emotion, noise_scale_w=morpheme,
                            length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    sf.write(output, audio, samplerate=hps.data.sampling_rate)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
