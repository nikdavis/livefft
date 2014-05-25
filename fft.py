import matplotlib
#required by OS X, must be done before other matplotlib imports
matplotlib.use('TKAgg')

from sys import byteorder
import array
from cmath import exp
from math import pi, log10
from matplotlib import animation
import time
import matplotlib.pyplot as plt
import numpy as np
import pylab

N = 2048
Fs = 44100
f = open('sig_8kHz.txt', 'r')

text = ""
for line in f:
	text += line

signal = [float(val) for val in text.split(" ")]
#print x
#x = [2.0, 14.0, 2.0, 14.0]

#based on wikipedia's FFT psuedocode, expects floats
def fft(x, N):
	fft.count += 1
	if N == 2:
		t = x[0]
		t2 = x[1]
		x[0] = t + t2
		x[1] = t - t2
		return x
	else:
		Nh = N / 2
		xe = x[0::2]
		xo = x[1::2]
		Xh0 = fft(xe, Nh)	# first half of FFT
		Xh1 = fft(xo, Nh)	# second half of FFT
		X = Xh0 + Xh1
	for k in range(Nh):
		t = X[k]
		t2 = X[k + Nh]
		e = exp(-2 * pi * 1j * k / N)
		X[k] = t + e * t2
		X[k + Nh] = t - e * t2
	return X
fft.count = 0


import pyaudio
import wave
import struct

def getChunks(lst, n):
	l = len(lst)
	for i in xrange(0, l, n):
		yield( lst[i:i+n] )


def rawToShort(raw, chs, ch):
	stride = 4		# 2 chars for short * 2 channels
	if ch != 0 and ch != 1:
		ch == 0
	x = []
	# this grabs a whole frame (left and right), and keeps one
	for frame in getChunks(raw, stride):
		x.append( struct.unpack("<h", frame[(ch * 2):(ch * 2) + 2])[0] )	# <h -> little-endian short
	return x


CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 44100 #sample rate

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer
  
#make plot
fig, ax = plt.subplots()
#ax.xlabel("Frequency (Hz)")
#ax.ylabel("Magnitude")
#line, = ax.plot(x, X)
#ax.set_yscale('log')
ax.set_ylim(-50, 90)
ax.set_xlim(0, 3500)
line, = ax.plot([], [], lw=1)
x = [float(x) / N / 2 * Fs for x in range(0, N/2) ]
#start 

def update(X):
	line.set_data(x, X)
	return line,


def data_gen():
	while True:
		try:
			micData = rawToShort(stream.read(CHUNK), 2, 0)		
			signal = [float(i) for i in micData]
			X = [ i / N for i in fft(signal, N)[0:N/2] ]
			X = [10 * log10 (abs(y) ** 2) for y in X]
		except IOError:
			X = [ 0 for x in range(N/2) ]
			print "Overflow"
		yield X

def init():
    line.set_data([], [])
    return line,
   
data_gen.i = 0

ani = animation.FuncAnimation(fig, update, data_gen, init, interval=10, blit=True)
plt.show()

# Audio & array are LE, will need to check for consistency on big-endian systems

stream.stop_stream()
stream.close()
p.terminate()