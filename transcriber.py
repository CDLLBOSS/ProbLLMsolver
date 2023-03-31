import matplotlib.pyplot as plt
import numpy as np
import torch
from pyannote.audio import Model
from pyannote.core import notebook, Segment, SlidingWindow
from pyannote.core import SlidingWindowFeature as SWF
from streamz import Stream
import pyaudio
import numpy as np
import requests
import tempfile

import time

class AudioBuffer:
    def __init__(self, rate=44100, chunk=1024, segment_length=5):
        self.rate = rate
        self.chunk = chunk
        self.segment_length = segment_length
        self.buffer = np.empty((0,), dtype=np.int16)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self.audio_callback)

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.buffer = np.concatenate((self.buffer, audio_data))
        return (in_data, pyaudio.paContinue)

    def segment(self, index):
        start = index * (self.rate * (self.segment_length - 1))
        end = start + (self.rate * self.segment_length)

        if end > len(self.buffer):
            return None

        return self.buffer[start:end]

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# Function to visualize the VAD result with moving average
def visualize(features):
    speech_prob = features.data[:, 0]
    window_size = 40
    avg_speech_prob = moving_average(speech_prob, window_size)
    
    threshold = 0.3  # You can adjust this value based on your requirements
    indices_below = cross_from_below(avg_speech_prob, threshold)
    indices_above = cross_from_above(avg_speech_prob, threshold)
    
    plt.figure(figsize=(12, 6))
    plt.plot(speech_prob, label='Speech Probability')
    plt.plot(np.arange(window_size - 21, len(speech_prob)-20), avg_speech_prob, label='Moving Average')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    for index in indices_below:
        plt.axvline(x=window_size - 21 + index, color='b', linestyle='--', linewidth=1)
    
    for index in indices_above:
        plt.axvline(x=window_size - 21 + index, color='r', linestyle='--', linewidth=1)
    
    plt.legend()
    plt.show()
    



# Voice activity detection class
import torch
import numpy as np
from pyannote.audio import Model

class AudioPlayback:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk

        self.p = pyaudio.PyAudio()

    def play(self, audio_data):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=self.rate,
                             output=True)
        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()

    def close(self):
        self.p.terminate()

class VoiceActivityDetection:
    
    def __init__(self):
        self.model = Model.from_pretrained("pyannote/segmentation", use_auth_token="")
        self.model.eval()
        
    def __call__(self, current_buffer: SWF) -> SWF:
        
        # we start by applying the model on the current buffer
        with torch.no_grad():
            waveform = torch.tensor(np.array(current_buffer.data).T, dtype=torch.float32)

            segmentation = self.model(waveform[np.newaxis]).numpy()[0]

        # temporal resolution of the output of the model
        resolution = self.model.introspection.frames
        
        # temporal shift to keep track of current buffer start time
        resolution = SlidingWindow(start=current_buffer.sliding_window.start, 
                                   duration=resolution.duration, 
                                   step=resolution.step)
            
        # pyannote/segmentation pretrained model actually does more than just voice activity detection
        # see https://huggingface.co/pyannote/segmentation for more details.     
        speech_probability = np.max(segmentation, axis=-1, keepdims=True)
        
        return SWF(speech_probability, resolution)
    
def cross_from_above(probabilities, threshold):
    crossing_indices = []
    for i in range(1, len(probabilities)):
        if probabilities[i - 1] > threshold and probabilities[i] <= threshold:
            crossing_indices.append(i)
    return crossing_indices

def cross_from_below(probabilities, threshold):
    crossing_indices = []
    for i in range(1, len(probabilities)):
        if probabilities[i - 1] <= threshold and probabilities[i] > threshold:
            crossing_indices.append(i)
    return crossing_indices



# Initialize voice activity detection
vad = VoiceActivityDetection()

# Set up the stream
source = Stream()
source.map(vad).sink(visualize)

# Start recording audio
audio_buffer = AudioBuffer()
audio_buffer.start()

def plot_waveforms(before, after, rate):
    time_before = np.arange(0, len(before)) / rate
    time_after = np.arange(0, len(after)) / rate

    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_before, before)
    plt.title("Waveform Before Trimming")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time_after, after)
    plt.title("Waveform After Trimming")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

import requests
import io
import time
import openai   
openai.api_key = ""
                           
import openai
import wave
import tempfile

# ...


#write function here to transcribe audio

import os
import uuid

def save_and_transcribe(trimmed_segment):
    # Save the trimmed audio segment to a temporary WAV file
    temp_wav_name = f"temp_{uuid.uuid4().hex}.wav"
    with wave.open(temp_wav_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(audio_buffer.rate)
        wav_file.writeframes(trimmed_segment.tobytes())

    # Transcribe the temporary WAV file using Whisper ASR API
    with open(temp_wav_name, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Remove the temporary WAV file
    os.remove(temp_wav_name)

    return transcript["text"]



def process_vad(segment, rate):
    window = SlidingWindow(duration=1 / rate, step=1 / rate)
    swf_segment = SWF(segment.reshape(-1, 1), window)
    segment_length = 4
    
    vad = VoiceActivityDetection()
    features = vad(swf_segment)
    speech_prob = features.data[:, 0]
    window_size = 40
    avg_speech_prob = moving_average(speech_prob, window_size)
    initial_offset = int(0.5 * rate)
    final_offset = int(-0.5 * rate)
    segment = segment[initial_offset:final_offset]
    #play the segment
    #audio = AudioPlayback() 
    #audio.play(segment)

    initial_offset = 61
    final_offset = -61

    avg_speech_prob = avg_speech_prob[initial_offset:final_offset]
    ##print(avg_speech_prob.shape)
    

    threshold = 0.1
    indices_below = cross_from_below(avg_speech_prob, threshold)
    indices_above = cross_from_above(avg_speech_prob, threshold)



    # Merge indices_below and indices_above with labels
    crossing_indices = [{'index': idx, 'label': 'below'} for idx in indices_below] + \
                       [{'index': idx, 'label': 'above'} for idx in indices_above]
    

    # Sort crossing_indices by index
    crossing_indices.sort(key=lambda x: x['index'])
    #print (crossing_indices)
    

    segments_to_keep = []

    start_time = 0
    end_time = 0

    if len(crossing_indices) != 0:
        if crossing_indices[0]['label'] == 'below':
            start_time = crossing_indices[0]['index'] / 773 * 4

        if crossing_indices[-1]['label'] == 'above':
            end_time = (773-crossing_indices[-1]['index']) / 773 * 4

    #print ("start: "+str(start_time),"end: "+str(end_time))

    if len(crossing_indices) != 0:

        for i, crossing in enumerate(crossing_indices):
            current_idx = int((crossing['index']) / 773 * rate * segment_length)

            if crossing['label'] == 'below':
                next_above_index = None

                # Find the next 'above' crossing index, if any
                for j, next_crossing in enumerate(crossing_indices[i + 1:]):
                    if next_crossing['label'] == 'above':
                        next_above_index = int((next_crossing['index']) / 773 * rate * segment_length)
                        del crossing_indices[i + 1 + j]  # Remove the 'above' index after saving it
                        break

                if next_above_index:
                    segments_to_keep.append(segment[current_idx:next_above_index])
                    #print ("start: "+str(current_idx),"end: "+str(next_above_index))
                else:
                    segments_to_keep.append(segment[current_idx:])
                    #print("start: " + str(current_idx), "end: " + str(len(segment)))

            elif crossing['label'] == 'above' and (i == 0 or crossing_indices[i - 1]['label'] != 'below'):
                segments_to_keep.append(segment[:current_idx])
                #print ("start: "+str(0),"end: "+str(current_idx))
        
    else:
        #get the average speech probability of the whole segment
        avg_speech_prob = np.mean(speech_prob)
        if avg_speech_prob >= 0.2:
            segments_to_keep.append(segment)
        if avg_speech_prob < 0.2:
            #print("no speech detected")
            return None, start_time, end_time
    # Concatenate all segments to keep
    trimmed_segment = np.concatenate(segments_to_keep)

    return trimmed_segment, start_time, end_time



i = 0

lastend = 0
start = 0

transcriptionlist = []

segmentstocombine = []
while True:


    segment = audio_buffer.segment(i)

    if segment is not None:
        window = SlidingWindow(duration=1 / audio_buffer.rate, step=1 / audio_buffer.rate)
        #show the waveform
        
        
        # swf_segment = SWF(segment.reshape(-1, 1), window)
        # source.emit(swf_segment)
        
        star = time.time()
        trimmed_segment, start, end = process_vad(segment, audio_buffer.rate)
        end = time.time()
        print(end-star)
        t = 4


        #AudioPlayback().play(trimmed_segment)
        if trimmed_segment is not None:
            if start != 0:
                space = start + lastend
            if start == 0:
                space = lastend
            lastend = end
            if space <= t:
                #print("space1: "+str(space))
                segmentstocombine.append(trimmed_segment)
            if space > t:
                #print("space2: "+str(space))
                if segmentstocombine != []:
                    segmentstocombine = np.concatenate(segmentstocombine)
                    #trans = save_and_transcribe(segmentstocombine)
                    #print(trans)
                    #transcriptionlist.append(trans)
                    AudioPlayback().play(segmentstocombine)
                    segmentstocombine = []
                    segmentstocombine.append(trimmed_segment)
        if trimmed_segment is None:
            #print("no speech detected")
            if segmentstocombine != []:
                segmentstocombine = np.concatenate(segmentstocombine)
                #trans = save_and_transcribe(segmentstocombine)
                #print(trans)
                #transcriptionlist.append(trans)
                AudioPlayback().play(segmentstocombine)
                segmentstocombine = []
            
        i = i + 1
        
        
    else:
        
        #print(f"Segment {i} not available")
        time.sleep(1)






