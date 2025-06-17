from comet_ml import Experiment
exp = Experiment(api_key="...", project_name="test-audio-upload")
from scipy.io.wavfile import write
import numpy as np

sr = 22050
samples = np.random.randn(sr * 2)
samples = np.int16(samples / np.max(np.abs(samples)) * 32767)

wav_fp = "test_audio.wav"
write(wav_fp, sr, samples)
exp.log_audio(wav_fp)
exp.end()
