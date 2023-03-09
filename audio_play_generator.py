#!/usr/bin/env python3


import os
import torch
import time
from IPython.display import Audio, display
import pygame

def user_input_name():
	user_input = ""
	while user_input == "":
		user_input = input("Type name model(or exit): ")
		if user_input == "exit":
			return "exit"
	return user_input+".pt"

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

ssml_sample = """
              <speak>
              <p>
                  <prosody rate="fast">Мал+ыш, ты, кажется не вступ+ил ещ+ё, в альянс. Гот+овишься к ох+оте, в один+очку?</prosody> Один из бр+атьев, осмотрел Чу Ф+эна с ног до голов+ы.
                  <p>
                  -С +этим. Чт+ото не так?" Не п+онял Чу Фэн.
                  </p>
                  <p>
                  <prosody pitch="high">-Б+уду честен. Эта ох+ота, не то, что м+ожет быть сд+елано в одиночку. Предлаг+аю теб+е, вступ+ить в альянс.</prosody> Два бр+ата, любезно, д+али ем+у совет.
                  </p>
                  <p>
                  </p>
              </p>
              </speak>
              """

sample_rate = 48000
speaker = 'random'
pygame.mixer.init()
audio_paths = model.apply_tts(text=ssml_sample,
								speaker=speaker,
								sample_rate=sample_rate,
								voice_path="girl.pt")
with open('newsound.wav', 'wb+') as f:
	f.write(Audio(audio_paths, rate=sample_rate).data)
pygame.mixer.music.load("newsound.wav")
pygame.mixer.music.play()
time.sleep(10)
