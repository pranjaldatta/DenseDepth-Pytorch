import os
import argparse as arg
import shutil

import torch

import numpy as np
import cv2
from PIL import Image
from glob import glob

from data import load_testloader
from model import DenseDepth
from utils import colorize, DepthNorm, AverageMeter, load_images
from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion


def main():

    parser = arg.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, help="path to checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument(
        "--data", type=str, default="examples/", help="Path to dataset zip file"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="plasma",
        help="Colormap to be used for the predictions",
    )

    args = parser.parse_args()

    if len(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("{} no such file".format(args.checkpoint))

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    print("Using device: {}".format(device))

    # Initializing the model and loading the pretrained model
    model = DenseDepth(encoder_pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=torch.device(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("model load from checkpoint complete ...")

    # Get Test Images
    img_list = glob(args.data + "*.png")

    # making processed image directory
    try:
        os.mkdir("examples/processed/")
    except FileExistsError:
        shutil.rmtree("examples/processed/")
        os.mkdir("examples/processed/")
        pass

    save_path = "examples/processed/"

    # Set model to eval mode
    model.eval()

    print(f"Number of images detected: {len(img_list)}")
    # Begin testing loop
    print("Begin Test Loop ...")

    for idx, img_name in enumerate(img_list):

        img = load_images([img_name])
        img = torch.Tensor(img).float().to(device)
        print("Processing {}, Tensor Shape: {}".format(img_name, img.shape))

        with torch.no_grad():
            preds = DepthNorm(model(img).squeeze(0))

        output = colorize(preds.data, cmap=args.cmap)
        output = output.transpose((1, 2, 0))
        cv2.imwrite(
            save_path + os.path.basename(img_name).split(".")[0] + "_result.png", output
        )

        print("Processing {} done.".format(img_name))


if __name__ == "__main__":
    print("Using torch version: ", torch.__version__)
    main()
