import numpy as np
from scipy.fft import dctn
from math import ceil
from utils import *

class Encoder:
    def __init__(self, compressionFactor) -> None:
        self.compressionFactor = compressionFactor
    
    def _rgb2ycbcr(self, im):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:,:,[1,2]] += 128
        return np.uint8(ycbcr)
    
    def _encodeBlock(self, block, quantization_type):
        if quantization_type == 'Y':
            Q_jpeg = 10 * np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 28, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])
        elif quantization_type == 'C':
            Q_jpeg = 10 * np.array([
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

        
    def encode(self, img, mse_bound = 5):
        X = img.copy()
        
        # ex3
            
        # ex2
        if len(X.shape) == 3:
            X = self._rgb2ycbcr(X)
            y, cb, cr    = self._padLayer(X[:, :, 0]), self._padLayer(X[:, :, 1]), self._padLayer(X[:, :, 2])
            (y, means_y), (cb, means_cb), (cr, means_cr) = self._compressLayer(y, 'Y'), self._compressLayer(cb, 'C'), self._compressLayer(cr, 'C')
            
            y = np.array([zigzag(b) for b in y]).astype(np.int16)
            cb = np.array([zigzag(b) for b in cb]).astype(np.int16)
            cr = np.array([zigzag(b) for b in cr]).astype(np.int16)
            
            yEncoded = run_length_encoding(y)
            yFrequencyTable = get_freq_dict(yEncoded)
            yHuffman = find_huffman(yFrequencyTable)
            
            crEncoded = run_length_encoding(cr)
            crFrequencyTable = get_freq_dict(crEncoded)
            crHuffman = find_huffman(crFrequencyTable)

            cbEncoded = run_length_encoding(cb)
            cbFrequencyTable = get_freq_dict(cbEncoded)
            cbHuffman = find_huffman(cbFrequencyTable)
            
            yBitsToTransmit = str()
            for value in yEncoded:
                yBitsToTransmit += yHuffman[value]

            crBitsToTransmit = str()
            for value in crEncoded:
                crBitsToTransmit += crHuffman[value]

            cbBitsToTransmit = str()
            for value in cbEncoded:
                cbBitsToTransmit += cbHuffman[value]
            
            means_y = " ".join(x for x in means_y)
            means_cb = " ".join(x for x in means_cb)
            means_cr = " ".join(x for x in means_cr)
            
            return str(X.shape[0]) + " " + str(X.shape[1]) + "\n".join([yBitsToTransmit, means_y, cbBitsToTransmit, means_cb, crBitsToTransmit, means_cr])
        elif len(X.shape) == 2:
            X = self._compressLayer(X, 'Y')
        #TODO HUFFMAN ENCODING
    
        return X