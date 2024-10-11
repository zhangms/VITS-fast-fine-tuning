import argparse

import torch

import utils
from models import SynthesizerTrn

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Predictor:
    def __init__(self, model_path, config_path):
        self.speed = 1.1
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./OUTPUT_MODEL/G_latest.pth")
    parser.add_argument("--config_path", default="./OUTPUT_MODEL/config.json")
    args = parser.parse_args()
    print("model_path:", args.model_path)
    print("config_path:", args.config_path)
    predictor = Predictor(args.model_path, args.config_path)
