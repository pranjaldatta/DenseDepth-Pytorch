import os  
import argparse as arg  
import time  

import torch   

import numpy as np  
import cv2   
from PIL import Image   

from data import load_testloader
from model import DenseDepth
from utils import colorize, DepthNorm, AverageMeter
from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion


def main():

    parser = arg.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, help="path to checkpoint")
    parser.add_argument("--savedir", "-s", type=str, default="", help="save location of the test images")
    parser.add_argument("--batch", "-b", type=int, default=1, help="batch size of test batchs")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="data/nyu_depth.zip", help="Path to dataset zip file")
    parser.add_argument("--theta", "-t", default=0.1, type=float, help="coeff for L1 (depth) Loss")

    args = parser.parse_args()

    if len(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("{} no such file".format(args.checkpoint))
    if args.batch != 1:
        raise ValueError("{} batch_size is not valid. Batch_size 1 is allowed".format(args.batch))
    
    # make image, depth, preds dirs
    if len(args.savedir) > 0  and not os.path.isdir(args.savedir):
        raise NotADirectoryError("{} not a directory".format(args.savedir))
    os.mkdir(args.savedir + "images/")
    os.mkdir(args.savedir + "depth/")
    os.mkdir(args.savedir + "preds/")

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    print("Using device: ".format(device))

    # Initializing the model and loading the pretrained model 
    model = DenseDepth(encoder_pretrained=False)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("model load from checkpoint complete ...")

    # Get TestLoader  
    print("Loading testloader ...")
    testloader = load_testloader(path)
    len_testloader = len(testloader)
    print("Tesloader load complete. len(testloader): {} ...".format(len_testloader))

    # Set model to eval mode 
    model.eval()

    # Begin testing loop 
    print("Begin Test Loop ...")
    l1_criterion = torch.nn.L1Loss()
    
    for idx, batch in enumerate(testloader):
        
        loss_meter = AverageMeter()

        with torch.no_grad():

            image_x = torch.Tensor(batch["image"]).to(device)
            depth_y = torch.Tensor(batch["depth"]).to(device)

            norm_depth_y = DepthNorm(depth_y)

            preds = model(image_x)
            
            # Loss functions
            l1_loss = l1_criterion(preds, norm_depth_y)
            ssim_loss = torch.clamp(
                (1 - ssim_criterion(preds, norm_depth_y))*0.5,
                min=0, 
                max=1
            )
            gradient_loss = gradient_criterion(norm_depth_y, preds, device=device)

            net_loss = (1.0 * ssim_loss) + (1.0 + gradient_loss) + (args.theta * l1_loss)

            loss_meter.update(net_loss.data.item(), image_x.size(0))

            print("batch idx: {}, loss:{}".format(idx, net_loss.data.item()))

            if idx % 3 == 0:
                
                save_preds = DepthNorm(preds.squeeze(0))
                save_preds = colorize(preds.data)
                name = "idx_{}.png"
                cv2_imwrite(args.savedir+"preds/"+name, save_preds)

                save_x = image_x.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite(args.savedir+"images/"+name, save_x)

                save_y = DepthNorm(depth_y.squeeze(0))
                save_y = colorize(save_y.data)
                cv2.imwrite(args.savedir+"preds/"+name, save_y)

                del save_preds
                del save_x
                del save_y







if __name__ == "__main__":
    print("Using torch version: ", torch.__version__)
    main()
