import numpy as np
from scipy.fft import dctn
from math import ceil
from utils import zigzag


class Encoder:
    def __init__(self, compressionFactor = 50) -> None:
        self.compressionFactor = compressionFactor
    
    def _rgb2ycbcr(self, im):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:,:,[1,2]] += 128
        return np.uint8(ycbcr)
    
    def _encodeBlock(self, block, quantization_type):
        # ex3
        if quantization_type == 'Y':
            Q_jpeg = self.compressionFactor * 10 ** 3 * np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 28, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])
        elif quantization_type == 'C':
            Q_jpeg = self.compressionFactor * 10 ** 3 * np.array([
                    [17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]])

        mean = np.mean(block)
        y = dctn(block - mean, s=(8, 8))
        y_jpeg = Q_jpeg*np.round(y/Q_jpeg)
        return (y_jpeg, mean)
        
    def _compressLayer(self, layer, quantization_type):
        X = layer.copy()
        h, w = X.shape 
        blocks, means = [], []
        for row in range(0, h, 8):
            for col in range(0, w, 8):
     
                block = X[row : row + 8, col : col + 8]
                blockEncoded, mean = self._encodeBlock(block, quantization_type)
                means.append(mean)
                blocks.append(blockEncoded)
        return blocks, means    
    
    def _padLayer(self, layer):
        result = None
        if (len(layer[0]) % 8 == 0) and (len(layer) % 8 == 0):
            result = layer.copy()
        else:
            w, l = ceil(len(layer[0]) / 8) * 8, ceil(len(layer) / 8) * 8
            layerPadded = np.zeros((l, w))
            for i in range(len(layer)):
                for j in range(len(layer[0])):
                    layerPadded[i, j] += layer[i, j]
            
            result = layerPadded
        
        return result
        
    def encode(self, img):
        
        # ex1
        
        X = img.copy()
        
        # ex2
        if len(X.shape) == 3:
            X = self._rgb2ycbcr(X)
            y, cb, cr    = self._padLayer(X[:, :, 0]), self._padLayer(X[:, :, 1]), self._padLayer(X[:, :, 2])
            (y, means_y), (cb, means_cb), (cr, means_cr) = self._compressLayer(y, 'Y'), self._compressLayer(cb, 'C'), self._compressLayer(cr, 'C')
            
            y = [zigzag(b) for b in y]
            cb = [zigzag(b) for b in cb]
            cr = [zigzag(b) for b in cr]
            
            y = np.array(y).astype(np.int16).flatten()
            cb = np.array(cb).astype(np.int16).flatten()
            cr = np.array(cr).astype(np.int16).flatten()
            
            yEncoded = " ".join([str(x) for x in y])
            crEncoded = " ".join([str(x) for x in cr])
            cbEncoded = " ".join([str(x) for x in cb])
            
            means_y = " ".join([str(x) for x in means_y])
            means_cb = " ".join([str(x) for x in means_cb])
            means_cr = " ".join([str(x) for x in means_cr])
            
            bitstream = str(X.shape[0]) + " " + str(X.shape[1]) + "\n" +  "\n".join([yEncoded, means_y, cbEncoded, means_cb, crEncoded, means_cr])
            bitstream = bitstream.encode()
            return bitstream
        elif len(X.shape) == 2:
            X = self._compressLayer(X, 'Y')
    
        return X