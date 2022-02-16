import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf

y_size = 10000
year = 10
r = np.random.randint(200, size=4)/100 + 3
p = np.random.randint(180, size=4)
t = np.linspace(0, year, y_size)
cutoff_power = 5000

f1 = 1
f2 = 3
f3 = 5

v_comp1 = r[0] * np.sin(2*np.pi*f1*t + p[0])
v_comp2 = r[1] * np.sin(2*np.pi*f2*t + p[1])
v_comp3 = r[2] * np.sin(2*np.pi*f3*t + p[2])
# v_accel = r[3] * np.sin(2*np.pi*0.01*t)
v_accel = 0.01*t

v_comp = v_comp1 + v_comp2 + v_comp3 + v_accel

plt.figure(figsize=(12,8))
plt.subplot(4, 1, 1)
plt.plot(t, v_comp1)
plt.ylabel("v_comp1",size=15)
plt.subplot(4, 1, 2)
plt.plot(t, v_comp2)
plt.ylabel("v_comp2",size=15)
plt.subplot(4, 1, 3)
plt.plot(t, v_comp3)
plt.ylabel("v_comp3",size=15)
plt.subplot(4, 1, 4)
plt.plot(t, v_comp)
plt.ylabel("v_comp",size=15)
plt.show()

#! ---------------------------------
#! Adding random noise..

v_real = v_comp + np.random.randn(y_size)

plt.figure(figsize=(12,2))
plt.plot(t, v_real)
plt.title('Noisy Signal')
plt.show()

# with open('./data.csv', 'w', encoding='UTF8', newline='') as file_open:
#     writer = csv.writer(file_open)
#     writer.writerow(['Time (t)', 'Observed Velocity'])
#     for i in range(0, len(v_real)):
#         writer.writerow([t[i], v_real[i]])

print(t)
print(v_real)

#! ----------------------------------
#! Fourier transform to find the frequency values..

power = np.fft.rfft(v_real)
freq = np.fft.rfftfreq(len(v_real), 0.001)
# print(freq)

# plt.ion()
plt.figure(figsize=(8,4))
plt.plot(freq, np.abs(power))
plt.show()

#! ----------------------------------
#! Denoising output fourier

indices=np.where(np.abs(power)<cutoff_power)
power[indices]=0
plt.figure(figsize=(8,4))
plt.plot(freq, np.abs(power))
plt.title("De-noised Power Spectrum")
plt.grid()
plt.show()


#The inverse functions takes in input the Fourier transformed array from our rfft function
inverse=np.fft.irfft(power) #Notice the i in irfft
# print(inverse.shape)

plt.figure(figsize=(12,2))
plt.plot(t, inverse)
plt.title("Inverse",size=15)
plt.xlabel("Time (t)",size=13)
plt.ylabel("Inverse",size=13)
plt.show()


#! Recovering linear const acc due to dark matter
sth = v_comp - inverse
plt.figure(figsize=(12,2))
plt.scatter(t, sth)
plt.show()

def eqn (t, m, c):
    return m*t+c
    

po, pc = cf(eqn, t, sth)
plt.plot(t, eqn(t, *po))
plt.show()