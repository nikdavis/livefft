import matplotlib
#required by OS X, must be done before other matplotlib imports
matplotlib.use('TKAgg')

from cmath import exp
from math import pi, log10
import matplotlib.pyplot as plt
from matplotlib import animation
import pyaudio
import struct
import sys

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
    Xh0 = fft(xe, Nh)  # first half of FFT
    Xh1 = fft(xo, Nh)  # second half of FFT
    X = Xh0 + Xh1
  for k in range(Nh):
    t = X[k]
    t2 = X[k + Nh]
    e = exp(-2 * pi * 1j * k / N)
    X[k] = t + e * t2
    X[k + Nh] = t - e * t2
  return X
fft.count = 0

def getChunks(lst, n):
  l = len(lst)
  for i in xrange(0, l, n):
    yield( lst[i:i+n] )

def rawToShort(raw, chs, ch):
  stride = 4    # 2 chars for short * 2 channels
  if ch != 0 and ch != 1:
    ch == 0
  x = []
  # this grabs a whole frame (left and right), and keeps one
  for frame in getChunks(raw, stride):
    x.append( struct.unpack("<h", frame[(ch * 2):(ch * 2) + 2])[0] )  # <h -> little-endian short
  return x

class DTMF_Detector:
  toneThresh = 45
  buttons = {}
  detected = 0
  detectedButton = ""
  
  def __init__(self, fftSize, sampFreq):
    self.x, self.y = self.DTMF_indices(fftSize, sampFreq)
  
  def buttonDetected(self, fftData):
    tones = self.tonesDetected(fftData)
    ret = None
    if tones:
      if self.detected == 0:
        self.detected = 1
        key = str(tones[0]) + ":" + str(tones[1])
        self.detectedButton = self.buttons[key]
      else:
        key = str(tones[0]) + ":" + str(tones[1])
        if self.detectedButton != self.buttons[key]:
          self.detected = 1
          btn = self.detectedButton
          self.detectedButton = self.buttons[key]
          ret = btn
    elif self.detected == 1:
      self.detected = 0;
      ret = self.detectedButton
    return ret

  def tonesDetected(self, fftData):
    f = fftData
    x = None
    y = None
    #print f[56]
    for idx, xi in enumerate(self.x):
      if f[xi] > self.toneThresh:
        x = idx
        break
    for idx, yi in enumerate(self.y):
      if f[yi] > self.toneThresh:
        y = idx
        break
    if x >= 0 and y >= 0:
      return (x, y)
    else:
      return None
      
  # Frequencies in Hz -- from top, left
  def DTMF_indices(self, fftSize, sampleFreq):
    # x:y
    self.buttons['0:0'] = '1'
    self.buttons['1:0'] = '2'
    self.buttons['2:0'] = '3'
    self.buttons['0:1'] = '4'
    self.buttons['1:1'] = '5'
    self.buttons['2:1'] = '6'
    self.buttons['0:2'] = '7'
    self.buttons['1:2'] = '8'
    self.buttons['2:2'] = '9'
    self.buttons['0:3'] = '*'
    self.buttons['1:3'] = '0'
    self.buttons['2:3'] = '#'
    x = []
    y = []
    x_ind = []
    y_ind = []
    x.append(1209.0)
    x.append(1336.0)
    x.append(1477.0)
    y.append(697.0)
    y.append(770.0)
    y.append(852.0)
    y.append(941.0)
    for freq in x:
      x_ind.append( int( round( freq / sampleFreq * fftSize ) ) )
    for freq in y:
      y_ind.append( int( round( freq / sampleFreq * fftSize ) ) )
    #print x_ind
    return x_ind, y_ind

N = 2048          # Chunk size in frames from mic
Fs = 44100          # Sample rate in Hz
toneThresh = 50
FORMAT = pyaudio.paInt16  # Format from mic
CHANNELS = 2

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=Fs,
                input=True,
                frames_per_buffer=N) #buffer
  
x = [float(x) / N * Fs for x in range(0, N/2) ]
#start 

d = DTMF_Detector(N, Fs)
while True:
  try:
    # could also use array('h', ..)
    micData = rawToShort(stream.read(N), 2, 0)    
    signal = [float(i) for i in micData]
    X_fft = [ i / N for i in fft(signal, N)[0:N/2] ]
    try:
      X = [10 * log10 (abs(y) ** 2) for y in X_fft]
    except ValueError:
      #print X_fft
      #print X
      None
  except IOError:
    X = [ 0 for x in range(N/2) ]
    print "Overflow"
  t = d.buttonDetected(X)
  if t:
    sys.stdout.write(t)
    sys.stdout.flush()
  
stream.stop_stream()
stream.close()
p.terminate()
