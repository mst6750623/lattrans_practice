import torch
import argparse
import os
import yaml
import torch.utils.data as data
from utils.dataset import LatentDataset
from trainer import *

parser = argparse.ArgumentParser(
    description="This is a description of trainer parser")
parser.add_argument("--config", type=str, default="001", required=True)
parser.add_argument("--ckpt", type=str, default='', help="checkpoint_path")
parser.add_argument("--latent_path",
                    type=str,
                    default="./data/celebahq_dlatents_psp.npy")
parser.add_argument("--latent_label",
                    type=str,
                    default="./data/celebahq_anno.npy")
parser.add_argument(
    "--stylegan_path",
    type=str,
    default="/mnt/pami23/stma/pretrained_models/psp_ffhq_encode.pt")


def main():
    args = parser.parse_args()
    config_path = "./config/" + args.config + ".yaml"
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    print(config)
    checkpoint_path = args.ckpt
    if checkpoint_path is not None:
        print("checkpoint_path is not none: ", checkpoint_path)

    device = config['device']
    batch_size = config['batch_size']
    epochs = config['epochs']
    attrs = config['attr'].split(',')
    classifier_path = "./models/latent_classifier_epoch_20.pth"
    latent_code = args.latent_path
    latent_label = args.latent_label
    dataset = LatentDataset(latent_code, latent_label, True)
    iter = data.DataLoader(dataset, batch_size, shuffle=True)
    for i, attr in enumerate(attrs):
        print("attr", i, ": ", attr)
        trainer = Trainer(i)
        trainer.initialize(args.stylegan_path, classifier_path)
        trainer.train()
        trainer.to(device)
        for epoch in range(epochs):
            print("in_epoch!")
            for id, list in enumerate(iter):
                code = list[0]
                label = list[1]
                code, label = code.to(device), label.to(device)
                print(code, label)
                if id > 10:
                    break


if __name__ == "__main__":
    main()