import torch  
import numpy as np  
import cv2  

from densedepth.losses import ssim


def show_img(img):

    while True:
        cv2.imshow("Frame", img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()

img = cv2.imread("tests/einstein.png")

# show initial picture 
show_img(img)

# generate random noise  
noise = torch.rand(img.shape)

# show random noise 
show_img(noise.numpy())

# get the ssim score between TRUE image and NOISE image 
img = img.transpose(2, 0, 1)
img = torch.tensor(img).unsqueeze(0) / 255.0
noise = noise.transpose(0, 2).transpose(1, 2).unsqueeze(0)


true_vs_noise = ssim(img, noise, 255)
#true_vs_noise = (1- true_vs_noise.data) / 2

# Now get ssim score between TRUE Image and TRUE Image 
true_vs_true = ssim(img, img, 255)
#true_vs_true = (1 - true_vs_true) / 2

print("True vs random Noise: {}\nTrue vs True:{}".format(true_vs_noise, true_vs_true))

if true_vs_true == 1.0 and true_vs_noise < 0.1: 
    print("Test successful")
else:
    print("Test failed")
