import matplotlib.pyplot as plt
import numpy as np

# ex1

# a)
start = 0 
end = 0.03 
step = 0.0005

timeSmoothSampled = np.arange(start, end, step)

# b)

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

# b)

fig, axs = plt.subplots(3)
fig.suptitle("Cele 3 semnale esantionate cu 2000 Hz")
fig.tight_layout()
axs[0].plot(timeSmoothSampled, x(timeSmoothSampled))
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('amplitude')
axs[1].plot(timeSmoothSampled, y(timeSmoothSampled))
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('amplitude')
axs[2].plot(timeSmoothSampled, z(timeSmoothSampled))
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('amplitude')

fig.show()

# c)

time = np.arange(start, end, 1 / 200)

fig, axs = plt.subplots(3)
fig.suptitle("Cele 3 semnale esantionate cu 200 Hz")
fig.tight_layout()

axs[0].stem(time, x(time))
axs[1].stem(time, y(time))
axs[2].stem(time, z(time))

axs[0].plot(timeSmoothSampled, x(timeSmoothSampled))
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('amplitude')
axs[1].plot(timeSmoothSampled, y(timeSmoothSampled))
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('amplitude')
axs[2].plot(timeSmoothSampled, z(timeSmoothSampled))
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('amplitude')

fig.show()


# ex2

plt.rcParams["figure.figsize"] = (30,10)

# a)

time1 = np.linspace(0, 0.01, 1600)
sine1 = np.sin(2 * np.pi * 400 * time1)

# b)

time2 = np.linspace(0, 3, 700)
sine2 = np.sin(2 * np.pi * 800 * time2)

# c)

timeSaw = np.arange(0, 0.1, 1 / 10000)
sawtooth = 240 * np.mod(timeSaw, 1 / 240)

# d)

timeSquare = np.arange(0, 0.1, 1 / 4600)
square = np.sign(np.sin(2 * np.pi * timeSquare * 300))

# e)

mat1 = np.random.rand(128, 128)

# f)

mat2 = np.ones((128, 128))
mat2[64 : 96, 64 : 96] = np.zeros((32, 32))

fig, axs = plt.subplots(4)
fig.suptitle("Cele 4 semnale in ordine a), b), c), d)")

plt.rcParams["figure.figsize"] = (30,10)

axs[0].plot(time1, sine1)
axs[1].plot(time2, sine2)
axs[2].plot(timeSaw, sawtooth)
axs[3].plot(timeSquare, square)

plt.show()

plt.imshow(mat1)
plt.title("Semnal e)")

plt.show()


plt.imshow(mat2)
plt.title("Semnal f)")

plt.show()


# ex3

# a) 1 / 2000 = 0.0005
# b) 1h = 3600s 
#     2000 esantioane/s * 3600s = 7200000 esantioane 
#     7200000 * 4 biti = 28800000 biti 
#     28800000 biti / 8 biti = 3600000 bytes 