########
#获取lable文件的lable列表
########

import os

def get_key(file):
	key_words = ''
	with open(file) as fp:
		line = fp.readline()
		while line:
			line_new = line.replace(' ', '')
			line_new = line_new.replace('\n', '')
			line_new = line_new.replace('\r', '')
			key_words = key_words + line_new
			line = fp.readline()
	return key_words

#key_words = get_key("/home/gunn/lstm_ctc_ocr_beta/lib/lstm/new_words")
#print key_words
