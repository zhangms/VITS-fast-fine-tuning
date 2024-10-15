import argparse
import os

import numpy as np
import torch
from pydub import AudioSegment
from scipy.io import wavfile
from torch import LongTensor, no_grad

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Predictor:
    def __init__(self, model_path, config_path):
        self.speed = 1.0
        self.hps = utils.get_hparams_from_file(config_path)
        print(self.hps)
        self.net_g = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(device)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(model_path, self.net_g, None)
        self.tts_fn_id("hello world", 0, "warmup.mp3")

    def get_text(self, text, is_symbol):
        text_norm = text_to_sequence(text, self.hps.symbols, [] if is_symbol else self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def predict(self, text, n_speaker_id, speed, language):
        if text is not None:
            text = "[" + language + "]" + text + "[" + language + "]"
        stn_tst = self.get_text(text, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([n_speaker_id]).to(device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio

    def predict_id(self, text, speaker_id, speed, language):
        n_speaker_id = self.hps["speakers"][speaker_id]
        return self.predict(text, n_speaker_id, speed, language)

    def tts_fn_id(self, text, n_speaker_id, output_path):
        audio = self.predict(text, n_speaker_id, self.speed, "EN")
        wav = output_path + ".wav"
        audio = np.int16(audio * self.max_wav_value())
        wavfile.write(wav, self.sampling_rate(), audio)
        audio = AudioSegment.from_wav(wav)
        os.remove(wav)
        audio.export(output_path, format="mp3")
        return "OK"

    def tts_fn(self, text, speaker_id, output_path):
        n_speaker_id = self.hps["speakers"][speaker_id]
        return self.tts_fn_id(text, n_speaker_id, output_path)

    def get_speakers(self):
        return self.hps["speakers"]

    def sampling_rate(self):
        return self.hps.data["sampling_rate"]

    def max_wav_value(self):
        return self.hps.data["max_wav_value"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./OUTPUT_MODEL/G_latest.pth")
    parser.add_argument("--config_path", default="./OUTPUT_MODEL/config.json")
    args = parser.parse_args()
    print("model_path:", args.model_path)
    print("config_path:", args.config_path)
    predictor = Predictor(args.model_path, args.config_path)

    print(predictor.hps["speakers"]["Binary"])
    print(predictor.hps["speakers"]["Dara"])

    try:
        ret = predictor.tts_fn("Hey there! I'm always down for a chat. What's on your mind?", "Binary",
                               "/workspace/VITS-fast-fine-tuning/OUTPUT/c13501f8-d53e-4b89-a335-bb54eb6e055f.mp3")
        print(ret)
    except Exception as e:
        raise e
