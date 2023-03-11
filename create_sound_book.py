#!/usr/bin/env python3

import os
import torch
import pygame
import sys
import random
from pydub import AudioSegment
from IPython.display import Audio
import argparse
import threading


def generate_save_audio(model, speaker, sample_rate, voice_path, i, ssml):
	if ssml != "0":
		speaked_text = model.apply_tts(ssml_text="<speak>"+text+"</speak>",
									 speaker=speaker,
									 sample_rate=sample_rate,
									 voice_path=gen_model)
	else:
		speaked_text = model.apply_tts(text=text,
									 speaker=speaker,
									 sample_rate=sample_rate,
									 voice_path=gen_model)
	save_audio(str(i)+".wav", speaked_text, sample_rate)

def ssml_text_cut(text, len_paragraph):
	text_out = []
	while text != "" or len(text) >= 2:
		tmp_text = text[:len_paragraph]
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
		out_sounds = out_sounds + AudioSegment.from_wav("sound/pause"+random.choice(["1_5s", "2s"])+".wav")
	return out_sounds


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
					help=\
"""
Спикеры: aidar baya xenia kseniya eugene. С использованием GEN_MODEL только random. Некоторые спикеры работают с ошибками, например aidar""")
parser.add_argument("-t", "--text", default="text.txt")
parser.add_argument("-m", "--model", default="model.pt")
parser.add_argument("-g", "--gen_model", default="")
parser.add_argument("-s", "--speaker", default="eugene")
parser.add_argument("-o", "--out_sound", default="sound.wav")
parser.add_argument("-l", "--ssml", action=argparse.BooleanOptionalAction, default="0")
all_args = parser.parse_args(sys.argv[1:])

text_file = all_args.text
model_file = all_args.model
speaker = all_args.speaker
gen_model = all_args.gen_model
out_sound = all_args.out_sound
ssml_flag = all_args.ssml

len_paragraph = 990

if gen_model == "":
	gen_model = model_file
else:
	speaker = "random"
	len_paragraph = 500

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

audio_files_names = []
device = torch.device('cpu')
torch.set_num_threads(4)

if not os.path.isfile(model_file):
	torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
									model_file)


model = torch.package.PackageImporter(model_file).load_pickle("tts_models", "model")
model.to(device)

ssml_sample = ssml_text_cut(ssml_sample, len_paragraph)
threads = []
thread_alive = True
for i in range(0, len(ssml_sample)):
	text = ssml_sample[i]
	if len(text) < 4:
		continue
	print(str(i)+") generate")
	print(text)
	threads.append(threading.Thread(target=generate_save_audio, args=[model, speaker, sample_rate, gen_model, i, ssml_flag]))
	threads[-1].start()
	audio_files_names.append(str(i)+".wav")

while thread_alive:
	all_alive = len(threads)
	for i in threads:
		if not i.is_alive():
			all_alive -= 1
	if all_alive == 0:
		thread_alive = False

split_audio(audio_files_names).export(out_sound, format="wav")
remove_files(audio_files_names)
