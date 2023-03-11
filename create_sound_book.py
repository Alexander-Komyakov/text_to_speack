#!/usr/bin/env python3

import os
import torch
import pygame
import sys
import random
from pydub import AudioSegment
from IPython.display import Audio


def ssml_text_cut(text):
	text_out = []
	while text != "" or len(text) >= 2:
		tmp_text = text[:990]
		pos_begin = tmp_text.find("<p>")
		pos_end = tmp_text.find("</p>")+4
		if pos_end == -1+4:
			pos_begin = 0
			pos_end = tmp_text.rfind("\n")
			if pos_end == -1:
				pos_end = tmp_text.rfind(" ")
				if pos_end == -1:
					tmp_text += "</p>"
					pos_end = len(tmp_text)
		pos_end = tmp_text.find("</p>")+4
		if pos_end == -1+4:
			tmp_text += "</p>"
			pos_end = len(tmp_text)
		text_out.append(tmp_text[pos_begin:pos_end])
		text = text[pos_end+1:]
	return text_out

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
			out_sounds = out_sounds + AudioSegment.from_wav("sound/pause"+random.choice(["1_5s", "2s"])+".wav")
			continue 
		out_sounds = out_sounds + AudioSegment.from_wav(audio_array[i])
		out_sounds = out_sounds + AudioSegment.from_wav("sound/pause"+random.choice(["1_5s", "2s", "3s"])+".wav")
	return out_sounds

text_file = sys.argv[1]
model_file = sys.argv[2]
voice_path = sys.argv[3]
if len(sys.argv) > 4:
	ssml_flag = sys.argv[4]
else:
	ssml_flag = 0

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

ssml_sample = ssml_text_cut(ssml_sample)
for i in range(0, len(ssml_sample)):
	text = ssml_sample[i]
	print(str(i)+") generate")
	print(text)
	if ssml_flag == "--ssml":
		audio_model = model.apply_tts(ssml_text="<speak>"+text+"</speak>",
									 speaker=speaker,
									 sample_rate=sample_rate,
									 voice_path=model_file)
	else:
		audio_model = model.apply_tts(text=text,
								 speaker=speaker,
								 sample_rate=sample_rate,
								 voice_path=model_file)
	save_audio(str(i)+".wav", audio_model, sample_rate)
	audio_files_names.append(str(i)+".wav")

split_audio(audio_files_names).export(voice_path, format="wav")
remove_files(audio_files_names)
