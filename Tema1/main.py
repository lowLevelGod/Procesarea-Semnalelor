from encoder import Encoder
from decoder import Decoder
from scipy import misc
import matplotlib.pyplot as plt
from huffman import *

if __name__ == "__main__":
    X = misc.face()

    encoder = Encoder()
    result = encoder.encode(X)
    
    compressed = compress(result)
    
    with open("output.txt", "wb") as f:
        f.write(compressed)
    
    decompress("output.txt", "input.txt")
    
    decoder = Decoder()
    X_jpeg = decoder.decode("input.txt")
    
    plt.subplot(121).imshow(X)
    plt.title('Original')
    plt.subplot(122).imshow(X_jpeg)
    plt.title('JPEG')
    plt.savefig("raccoon.pdf")
    plt.show()
    
    # ex4
    import cv2

    def compressVideo(video):
        success,image = video.read()
        count = 0
        while success:
            encoder = Encoder()
            result = encoder.encode(image)
            
            compressed = compress(result)
            
            with open("output.txt", "wb") as f:
                f.write(compressed)
            
            decompress("output.txt", "input.txt")
            
            decoder = Decoder()
            image_jpeg = decoder.decode("input.txt")
            
            plt.subplot(121).imshow(image)
            plt.title('Original')
            plt.subplot(122).imshow(image_jpeg)
            plt.title('JPEG')
            plt.savefig("video_frame" + str(count) + ".pdf")
            plt.show()
            success,image = video.read()
            count += 1
        
    video = cv2.VideoCapture('BigBuckBunnyTrim.mp4')

    compressVideo(video)