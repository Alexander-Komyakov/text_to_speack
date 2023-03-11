#!/bin/python


import sys
import os
import argparse
import re



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--text", default="text.txt")
	parser.add_argument("-d", "--dictonary", default="words.txt")
	all_args = parser.parse_args(sys.argv[1:])

	text_file = all_args.text
	dictonary = all_args.dictonary

	dictonary_all = ""
	if os.path.isfile(dictonary):
		dictonary_all = read_all_file(dictonary)
	elif os.path.isdir(dictonary):
		for i in os.listdir(dictonary):
			dictonary_all += read_all_file(dictonary+"/"+i)
	else:
		dictonary_all = dictonary+" "

	dictonary_all = dictonary_all[:-1]

	if os.path.isfile(text_file):
		text = read_all_file(text_file)
		write_to_file(path=text_file, text=replace_word(text, dictonary_all))
	elif os.path.isdir(text):
		for i in os.listdir(text):
			path_to_text = text_file+"/"+i
			write_to_file(path_to_text,
							replace_word(read_all_file(path_to_text), dictonary_all))
	else: 
		print("NOT FOUND YOUR FILE OR DIRECTORY: ", text_file)

	
def replace_word(text, dictonary):
	symbols_reg = ("\s", "^", "\.", "\,", "\:", "\-", "\n", '\"', "\'")
	symbols = (" ", "", ".", ",", ":", "-", "\n", '"', "'")
	text_out = text
	for i in dictonary.split("\n"):
		source, final = i.split(" ")[0], i.split(" ")[1]
		for j in range(0, len(symbols_reg)):
			for k in range(0, len(symbols_reg)):
				text_out = re.sub(symbols_reg[j]+source+symbols_reg[k],
								symbols[j]+final+symbols[k],
								text_out)
	return text_out

def read_all_file(text_file):
	with open(text_file, "r") as f:
		text = f.read()
	return text

def write_to_file(path, text):
	with open(path, "w") as f:
		f.write(text)
	return 0

main()
