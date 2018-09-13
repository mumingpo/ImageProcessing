import numpy as np
from PIL import Image as img
import sys

def main():
    # read file
    if len(sys.argv) == 1:
        path1 = r"C:\Users\mumin\Desktop\1.png"
        path2 = r"C:\Users\mumin\Desktop\2.png"
        path3 = r"C:\Users\mumin\Desktop\3.png"
    else:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
        path3 = sys.argv[3]
    if len(sys.argv) == 5:
      bla = float(sys.argv[4])
    else:
      bla = 0.333333
    #p1_ = img.open(path1).convert("L")
    p2 = img.open(path2).convert("L")
    p1_ = img.open(path1).convert("L")
    p1_.thumbnail(p2.size)
    p1 = img.new("L", p2.size, color=255)
    upperleft = ((p2.size[0] - p1_.size[0]) // 2,
                (p2.size[1] - p1_.size[1]) // 2)
    p1 .paste(p1_, upperleft)
    #resize p1 to match size of p2
    #p1array = np.full(p2.size[::-1], 255, dtype=np.uint8)

    #hratio = p1_.size[0] / p2.size[0]
    #vratio = p1_.size[1] / p2.size[1]
    
    #if hratio >= vratio:
    #    p1_ = p1_.resize([p2.size[0], int(p1_.size[1] / hratio)])
    #    vstart = (p2.size[1]-p1_.size[1])//2
    #    p1array[vstart:vstart+p1_.size[1],:] = np.array(p1_)
    #else:
    #    p1_ = p1_.resize([int(p1_.size[0] / vratio), p2.size[1]])
    #    hstart = (p2.size[0]-p1_.size[0])//2
    #    p1array[:,hstart:hstart+p1_.size[0]] = np.array(p1_)
    #p1 = img.fromarray(p1array, "L")
    v, a = map(img.fromarray, GetVAPair(np.array(p1), np.array(p2), bla))
    p = img.merge("LA", [v,a])
    p.save(path3)

def GetVAPair(p1:np.ndarray, p2:np.ndarray, bla = 0.333333):    # back light alpha is the maximum brightness for the hidden image and minimum brightness for the dominant image
    vaproduct = bla * p2 / 255
    alpha = vaproduct + (1 - p1 / 255) * (1 - bla)
    value = np.divide(vaproduct, alpha, out=np.zeros_like(vaproduct), where=alpha!=0)
    return (value*255).astype(np.uint8), (alpha*255).astype(np.uint8)

if __name__ == "__main__":
    main()