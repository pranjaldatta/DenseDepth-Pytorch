import numpy as  np   
import matplotlib
import matplotlib.cm as cm

def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth

class AverageMeter(object):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count  

def colorize(value, vmin=10, vmax=1000, cmap="plasma"):

    value = value.cpu().numpy()[0, :, :]

    # normalize 
    vmin = value.min() if vmin is None else vmin 
    vmax = value.max() if vmax is None else vmax   
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0  
    
    cmapper =  cm.get_cmap(cmap)
    value = cmapper(value, bytes=True) 

    img = value[:,:,:3]

    return img.transpose((2, 0, 1))