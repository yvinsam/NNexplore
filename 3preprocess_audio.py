'''
All the stuff needed to preprocess audio data for NN
'''
import librosa, librosa.display
import matplotlib.pyplot as plt 
import os
import numpy as np

data_folder = os.path.join(os.getcwd(), '../GuitarNotes/')
print(data_folder)
file='choice'
#sound src filepaths:
## ../pitch/sms-tools/sounds/
## ./GuitarNotes/

#waveform
signal, sr = librosa.load(librosa.ex(file), sr=22050) #sr * duration(T) --> 22050 * 25
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

#fft --> spectrum
'''
FFT:
- Moves the signal from time to frequency domain
- No time information
- Static snapshot of amplitude and frequency for the entire duration
'''
fft = np.fft.fft(signal) 

magnitude = np.abs(fft) #gives us the magnitude of frequency (and converts from complex plane)
frequency = np.linspace(0, sr, len(magnitude))
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

#stft --> spectrogram
''' STFT:
- Computes several FFT at different intervals
- Preserves time info
- Fixed frame size( ex: 2048 samples)
- Gives a spectrogram (time + freq + mag)
'''
n_fft = 2048 #window for fft
hop_length = 512 
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

# log_spectrogram = librosa.amplitude_to_db(spectrogram)
# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCCs
'''
Mel Frequency Cepstral Coefficients
- Capture the timbral/textural aspects of sound
- Frequency domain feature
- Approximate human auditory system
- 13 to 40 coefficients
- Calculated each frame

'''
MFCCs = librosa.feature.mfcc(signal,n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()




