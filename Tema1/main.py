from encoder import Encoder
from decoder import Decoder
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


if __name__ == "__main__":
    X = misc.face()

    encoder = Encoder(None)
    decoder = Decoder()
     
    X_jpeg = decoder.decode(encoder.encode(X))
    
    plt.subplot(121).imshow(X)
    plt.title('Original')
    plt.subplot(122).imshow(X_jpeg)
    plt.title('JPEG')
    plt.savefig("raccoon.pdf")
    plt.show()