import numpy as np
import matplotlib.pyplot as plt
import time

# ex1

def dft(N, x):
    w = np.e ** (-2 * np.pi * 1j / N)

    fourierMatrix = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            fourierMatrix[i, j] = w ** (i * j)
            
    return np.dot(fourierMatrix, x)

time_dft = []
time_fft = []
for N in [128, 256, 512, 1024, 2048, 4096, 8192]:
    sine = np.sin(2 * np.pi * np.linspace(0, 1, N))
    
    start_dft = time.time()
    dft(N, sine)
    end_dft = time.time()
    
    time_dft.append(end_dft - start_dft)
    
    start_fft = time.time()
    np.fft.fft(sine)
    end_fft = time.time()
    
    time_fft.append(end_fft - start_fft)


plt.plot([128, 256, 512, 1024, 2048, 4096, 8192], np.log(time_dft), label="DFT O(N^2)")
plt.plot([128, 256, 512, 1024, 2048, 4096, 8192], np.log(time_fft), label="FFT O(NlogN)")
plt.legend()

plt.savefig("dft_vs_fft.pdf")
plt.show()