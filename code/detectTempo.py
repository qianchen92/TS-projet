import math as math
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt

#read the file
(rate,data)=wav.read('./wav/signal46.wav')

print(rate)
#filter bank
filter200n,filter200d = signal.ellip(6,3,40,200/rate*2,'lowpass')
filter200400n,filter200400d = signal.ellip(6,3,40,[200/rate*2,400/rate*2],'bandpass')
filter400800n,filter400800d = signal.ellip(6,3,40,[400/rate*2,800/rate*2],'bandpass')
filter8001600n,filter8001600d = signal.ellip(6,3,40,[800/rate*2,1600/rate*2],'bandpass')
filter16003200n,filter16003200d = signal.ellip(6,3,40,[1600/rate*2,3200/rate*2],'bandpass')
filter3200n,filter3200d = signal.ellip(6,3,40,3200/rate*2,'highpass')


#show the filter200
"""
w, h = signal.freqz(filter200400n, filter200400d)
plt.plot(w, 20 * np.log10(abs(h)))
#plt.plot(w,h)
plt.title("filter200")
plt.xlabel("frequency")
plt.ylabel("dB")
plt.show()
"""

#test for stability
'''
print ("filter1", np.all(np.abs(np.roots(filter200d))<1))
'''

signal200 = signal.lfilter(filter200n,filter200d,data,0)
signal200400 = signal.lfilter(filter200400n,filter200400d,data,0)
signal400800 = signal.lfilter(filter400800n,filter400800d,data,0)
signal8001600 = signal.lfilter(filter8001600n,filter8001600d,data,0)
signal16003200 = signal.lfilter(filter16003200n,filter16003200d,data,0)
signal3200 = signal.lfilter(filter3200n,filter3200d,data,0)


signalFilter = [signal200,signal200400,signal400800,signal8001600,signal16003200,signal3200]


#show the signal after filter
"""
t = signal200
sp = np.fft.fft(t)
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real)
plt.show()
"""

#envelope extraction
window = signal.hann(6400)
halfHann = window[3200:]

convolution = [0,0,0,0,0,0]

#convolution
for i in range(6):
	convolution[i] = signal.fftconvolve(signalFilter[i],halfHann)


#derivation
for j in range(6):
	for i in range(convolution[j].shape[-1]):
		convolution[j][i]  = (convolution[j][i]- convolution[j][i-1])

#rectify the signal
for i in range(6):
	convolution[i] = (np.absolute(convolution[i]) + convolution[i])/2


#display the Hann window
'''
plt.plot(halfHann)
plt.title("Hann window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()
'''

alpha = 0.5

def combFilter(tempo,x):
	length = x.shape[-1]
	T = math.floor(60*rate/tempo)
	comb = [1]*length
	for i in range(length):
		if i<T:
			comb[i]=(1-alpha)*x[i]
		else :
			comb[i]=comb[i-T]*alpha+(1-alpha)*x[i]
	return comb


tempoTab = range(40,180)
energieTempo = [[1]*140,[1]*140,[1]*140,[1]*140,[1]*140,[1]*140]

def energy (x) :
	somme = 0
	for i in x :
		somme = somme + i*i
	return somme

for i in range(6):
	for tempo in range(40,180):
		comb = combFilter(tempo,convolution[i])
		energieTempo[i][tempo-40] = energy(comb)

for i in range(6):
	plt.plot(tempoTab,energieTempo[i])
	plt.show()
