import numpy as np
import matplotlib.pyplot as plt

# ex1 

N = 8
w = np.e ** (-2 * np.pi * 1j / N)

fourierMatrix = np.zeros((N, N), dtype=complex)

for i in range(N):
    for j in range(N):
        fourierMatrix[i, j] = w ** (i * j)
        
fig, axs = plt.subplots(N, 2, figsize=(15, 15))
fig.suptitle("Fourier parte reala si imaginara per linie")

for i in range(N):
    real = [x.real for x in fourierMatrix[i]]
    imag = [x.imag for x in fourierMatrix[i]]
    
    axs[i][0].plot(range(N), real)
    axs[i][1].plot(range(N), imag)
    
print(np.linalg.norm(np.abs(np.matmul(fourierMatrix, fourierMatrix.conj().T) - N * np.identity(N))))

fig.savefig("ex1.pdf") 
fig.show()


# ex2

N = 1000

time = np.linspace(0, 1, N)
sine = np.sin(2 * np.pi * 7 * time + np.pi / 2)

y = sine * np.array([np.exp(-2 * np.pi * 1j * n / N) for n in range(N)])

plt.plot(range(N), sine, color='green')
plt.scatter([620], sine[620], color='red')
plt.title("Sinusoida f=" + str(7))
plt.savefig("ex2-sinusoida.pdf") 
plt.show()

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.scatter(y.real, y.imag, c=y.real ** 2 + y.imag ** 2)
plt.title("Cerc unitate")
plt.scatter(y.real[620], y.imag[620], color='red')
plt.savefig("ex2-cerc.pdf") 
plt.show()

z = lambda w: sine * np.array([np.e ** (-2 * np.pi * 1j * n * w / N) for n in range(N)])

for w in [1, 3, 5, 7]:
    
    x = z(w).real
    y = z(w).imag
    
    color = x**2 + y**2
    
    plt.scatter(z(w).real, z(w).imag, c=color)
    
    plt.scatter(np.mean(z(w).real), np.mean(z(w).imag), color="black")
    plt.plot([0, np.mean(z(w).real)], [0, np.mean(z(w).imag)], color="red", linewidth="2")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Cerc unitate w=" + str(w))
    plt.savefig("ex2-cerc_w_" + str(w) + ".pdf") 
    plt.show()
    
    
# ex3

def loopFourier(x, w):
    N = x.shape[0]
    e = np.array([np.e ** (-2 * np.pi * w * n * 1j / N) for n in range(N)])
    
    return np.sum(x * e)

N = 300
time = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * 10 * time) + 2 * np.sin(2 * np.pi * 35 * time) + 0.5 * np.sin(2 * np.pi * 60 * time)

fs = N / 60

fourier1 = []
freqDomain = fs * np.array(range(60))
for w in freqDomain:
    fourier1.append(loopFourier(x, w))

fourier1 = np.array(fourier1)

fig, axs = plt.subplots(2)
axs[0].plot(time, x)
axs[1].stem(freqDomain, fourier1)
fig.suptitle("Domeniul timp si domeniul frecventa")
plt.savefig("ex3.pdf") 
fig.show()