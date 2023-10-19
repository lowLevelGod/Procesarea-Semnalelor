import numpy as np
import matplotlib.pyplot as plt

# ex1

time = np.linspace(0, 1, 1000)
sine = 1 * np.sin(2 * np.pi * 1 * time + 0)
cosine = 1 * np.cos(2 * np.pi * 1 * time - np.pi / 2)


fig, axs = plt.subplots(2)
fig.suptitle("Cosine and sine waves")
axs[0].plot(time, sine)
axs[1].plot(time, cosine)

fig.show()

# ex2 

time = np.linspace(0, 1, 1000)

sine = 1 * np.sin(2 * np.pi * 1 * time + 0)
sine1 = 1 * np.sin(2 * np.pi * 1 * time + np.pi / 4)
sine2 = 1 * np.sin(2 * np.pi * 1 * time + np.pi / 2)
sine3 = 1 * np.sin(2 * np.pi * 1 * time + np.pi)

noise = np.random.normal(0, 1, 1000)
noise1 = np.random.normal(0, 1, 1000)
noise2 = np.random.normal(0, 1, 1000)
noise3 = np.random.normal(0, 1, 1000)

def computeGamma(x, z, snr):
    znorm = np.sum(np.square(z))
    xnorm = np.sum(np.square(x))
    
    return np.sqrt(xnorm / (snr * znorm))

noisySine = sine + computeGamma(sine, noise, 0.1) * noise
noisySine1 = sine1 + computeGamma(sine1, noise1, 1) * noise1
noisySine2 = sine2 + computeGamma(sine2, noise2, 10) * noise2
noisySine3 = sine3 + computeGamma(sine3, noise3, 100) * noise3


plt.title("Noisy sine waves")
plt.plot(time, noisySine, label="SNR 0.1")
plt.plot(time, noisySine1, label="SNR 1")
plt.plot(time, noisySine2, label="SNR 10")
plt.plot(time, noisySine3, label="SNR 100")

plt.legend()
plt.show()

# ex3

import sounddevice
import scipy

time1 = np.linspace(0, 0.01, 1600)
sine1 = np.sin(2 * np.pi * 400 * time1)

time2 = np.linspace(0, 3, 700)
sine2 = np.sin(2 * np.pi * 800 * time2)

timeSaw = np.arange(0, 0.1, 1 / 10000)
sawtooth = 240 * np.mod(timeSaw, 1 / 240)

timeSquare = np.arange(0, 0.1, 1 / 4600)
square = np.sign(np.sin(2 * np.pi * timeSquare * 300))

# sounddevice.play(sine1, 44100)
# sounddevice.play(sine2, 44100)
# sounddevice.play(sawtooth, 44100)
# sounddevice.play(square, 44100)

rate = int(10e5)
scipy.io.wavfile.write("sawtooth.wav", rate, sawtooth)
rate, x = scipy.io.wavfile.read("sawtooth.wav")

# ex4

time = np.linspace(0, 1, 1000)

sine = 1 * np.sin(2 * np.pi * 1 * time + 0)
sawtooth = 100 * np.mod(timeSaw, 1 / 100)

fig, axs = plt.subplots(3)
fig.suptitle("Sum of sine and sawtooth")
axs[0].plot(time, sine)
axs[1].plot(time, sawtooth)
axs[2].plot(time, sine + sawtooth)

fig.show()

# ex5

time = np.linspace(0, 1, 1000)

sine = 1 * np.sin(2 * np.pi * 1 * time + 0)
sine1 = 1 * np.sin(2 * np.pi * 10 * time + 0)

combined = np.concatenate([sine, sine1])

# sounddevice.play(combined, 44100)
# dupa o secunda sunetul este diferit pentru ca incepe a doua sinusoida
# iar sunetul este mai ascutit din cauza frecventei mai mari

def x(f, time):
    return np.sin(2 * np.pi * f * time)


time = np.linspace(0, 10, 100000000)
f0 = 10000
timeChirpExponential = 100 ** time

chirp = x(f0, timeChirpExponential)

sounddevice.play(chirp, 44100)

# ex6

time = np.linspace(0, 1, 12)
fs = 12

sinea = 1 * np.sin(2 * np.pi * (fs / 2) * time + 0)
sineb = 1 * np.sin(2 * np.pi * (fs / 4) * time + 0)
sinec = 1 * np.sin(2 * np.pi *     0    * time + 0)

fig, axs = plt.subplots(3)
fig.suptitle("Sine waves with different fundamental frequencies")
axs[0].plot(time, sinea, label="fs / 2", color="red")
axs[1].plot(time, sineb, label="fs / 4",  color="green")
axs[2].plot(time, sinec, label="0")

fig.legend()
fig.show()

# frecventa 0 este ne da practic sirul 0, 0, 0, 0 etc.
# cu cat frecventa de esantionare este mai mare, cu atat
# obtinem o sinusoida mai neteda

# ex7

# a)

fs = 20

time = np.linspace(0, 1, fs)
timeDecimated = np.linspace(0, 1, fs)[0::4]
timeDecimatedShifted =  np.linspace(0, 1, fs)[1::4]

sine = 1 * np.sin(2 * np.pi * 2 * time + 0)
sineDecimated = 1 * np.sin(2 * np.pi * 2 * timeDecimated + 0)
sineDecimatedShifted = 1 * np.sin(2 * np.pi * 2 * timeDecimatedShifted + 0)

plt.title("Decimated sine waves")
plt.plot(time, sine, label="fs")
plt.plot(timeDecimated, sineDecimated, label="fs / 4")
plt.plot(timeDecimatedShifted, sineDecimatedShifted, label="fs / 4 shifted")

plt.legend()
plt.show()

# o frecventa de esantionare mai mica (decimata) ne da o sinusoida mai putin neteda
# daca incepem cu al doilea element, practic shiftam sinusoida la stanga

# ex8

time = np.linspace(-np.pi / 2, np.pi / 2, 1000)
sine = np.sin(time)

taylorApprox = time 
padeApprox = (time - (7 * time ** 3) / 60) / (1 + time ** 2 / 20)

fig, axs = plt.subplots(2)
fig.suptitle("Sine approximations")
axs[0].plot(time, sine, label="sine", color="red")
axs[0].plot(time, taylorApprox, label="taylor",  color="green")
axs[0].plot(time, padeApprox, label="pade",  color="blue")
axs[1].semilogy(time, (np.abs(sine - taylorApprox)), label="error taylor")
axs[1].semilogy(time, (np.abs(sine - padeApprox)), label="error pade")


axs[0].legend()
axs[1].legend()
fig.show()
