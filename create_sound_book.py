#!/usr/bin/env python3

import os
import torch
import pygame
import sys
from pydub import AudioSegment
from IPython.display import Audio


def save_audio(name, audio, rate):
	with open(name, 'wb+') as f:
		f.write(Audio(audio, rate=rate).data)

def remove_files(ar):
	for i in ar:
		os.remove(i)

def split_audio(audio_array):
	for i in range(0, len(audio_array)):
		if i == 0:
			out_sounds = AudioSegment.from_wav(audio_array[i])
			continue 
		out_sounds = out_sounds + AudioSegment.from_wav(audio_array[i])
	return out_sounds

text_file = sys.argv[1]
model_file = sys.argv[2]
voice_path = sys.argv[3]

if os.path.isdir(text_file):
	ssml_sample = ""
	for i in os.listdir(text_file):
		with open(text_file+"/"+i, "r") as f:
			ssml_sample = ssml_sample + f.read()
elif os.path.isfile(text_file):
	with open(text_file, "r") as f:
		ssml_sample = f.read()
else:
	print("NOT FOUND ", text_file)
	sys.exit()

sample_rate = 48000
standart_speaker = ["aidar", "baya", "xenia", "kseniya", "eugene"]
speaker = 'random'
if model_file in standart_speaker:
	speaker = model_file

audio_files_names = []
len_text = 500
device = torch.device('cpu')
torch.set_num_threads(4)

if not os.path.isfile("model.pt"):
	torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
									"model.pt")


model = torch.package.PackageImporter("model.pt").load_pickle("tts_models", "model")
model.to(device)

for i in range(0, int(len(ssml_sample)/len_text)):
	print(str(i)+") generate")
	text_to_torch = ssml_sample[i*len_text:i*len_text+len_text]
	audio_model = model.apply_tts(text=text_to_torch,
								 speaker=speaker,
								 sample_rate=sample_rate,
								 voice_path=model_file)
	save_audio(str(i)+".wav", audio_model, sample_rate)
	audio_files_names.append(str(i)+".wav")

split_audio(audio_files_names).export(voice_path, format="wav")
remove_files(audio_files_names)
