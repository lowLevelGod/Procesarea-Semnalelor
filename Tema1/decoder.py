import numpy as np
from scipy.fft import idctn

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
               
    def decode(self, img):
        y, means_y, cb, means_cb, cr, means_cr = img 
        
        y = np.array([idctn(b) + m for (b, m) in zip(y, means_y)])
        y = np.block([b for b in y])
        
        print(y.shape)
        
        # X = img.copy()
        # # ex2
        # if len(X.shape) == 3:
        #     X = self._ycbcr2rgb(X)
        # elif len(X.shape) == 2:
        #     # X = compressLayer(X, 'Y')
        #     pass
        
        return X