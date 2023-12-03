import numpy as np
from scipy.fft import idctn
from math import ceil
from zigzag import inverse_zigzag

class Decoder:
    def __init__(self) -> None:
        pass
          
    def _ycbcr2rgb(self, im):
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(np.float64)
        rgb[:,:,[1,2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return np.uint8(rgb)
    
    def _restoreimg(self, array, h, w):
        
        img = np.zeros((h, w))
        idx = 0
        for row in range(0, h, 8):
            for col in range(0, w, 8):
        
                img[row : row + 8, col : col + 8] = array[idx]
                idx += 1
                
        return img

               
    def decode(self, file):
        with open(file, "r") as f:
            
            h, w = [int(x) for x in f.readline().strip().split()]
            paddedH, paddedW = ceil(h / 8) * 8, ceil(w / 8) * 8
                
            y = np.array([int(x) for x in f.readline().strip().split()])
            means_y = np.array([float(x) for x in f.readline().strip().split()])
            y = y.reshape((len(y) // 64, 64))
            
            cb = np.array([int(x) for x in f.readline().strip().split()])
            means_cb = np.array([float(x) for x in f.readline().strip().split()])
            cb = cb.reshape((len(cb) // 64, 64))
            
            cr = np.array([int(x) for x in f.readline().strip().split()])
            means_cr = np.array([float(x) for x in f.readline().strip().split()])
            cr = cr.reshape((len(cr) // 64, 64))
                    
            y = np.array([idctn(inverse_zigzag(b, 8, 8)) + m for (b, m) in zip(y, means_y)])
            cb = np.array([idctn(inverse_zigzag(b, 8, 8)) + m for (b, m) in zip(cb, means_cb)])
            cr = np.array([idctn(inverse_zigzag(b, 8, 8)) + m for (b, m) in zip(cr, means_cr)])
            
            y = self._restoreimg(y, paddedH, paddedW)
            cb = self._restoreimg(cb, paddedH, paddedW)
            cr = self._restoreimg(cr, paddedH, paddedW)
            
            y = np.round(y)[: h, : w].astype(np.int16)
            cb = np.round(cb)[: h, : w].astype(np.int16)
            cr = np.round(cr)[: h, : w].astype(np.int16)
                        
            # ex2
            X = np.dstack((y, cb, cr))
            if len(X.shape) == 3:
                X = self._ycbcr2rgb(X)
            
            return X