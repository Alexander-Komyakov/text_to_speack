#!/usr/bin/env python3


import os
import torch
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

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)

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

pygame.mixer.init()
while True:
	sample_rate = 48000
	speaker = 'random'
	put_accent=True
	put_yo=True

	audio_paths = model.save_wav(ssml_text=ssml_sample,
								 speaker=speaker,
								 sample_rate=sample_rate,
								 put_accent=put_accent,
								 put_yo=put_yo)

	#display(Audio("test.wav", sample_rate))
	pygame.mixer.music.load("test.wav")
	pygame.mixer.music.play()
	model_name = user_input_name()
	if model_name == "exit":
		break
	model.save_random_voice(model_name)
