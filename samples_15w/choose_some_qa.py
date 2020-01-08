# -*- coding: utf-8 -*-
# @Author: zhr1030635594
# @Date:   2019-11-21 00:58:22
# @Last Modified by:   zhr1030635594
# @Last Modified time: 2019-12-11 18:04:36
import os
import random
import numpy as np 
from opencc import OpenCC 

cc = OpenCC("t2s")

f = open("xhj1.conv", "r", encoding="utf-8")
fq = open("question2_15w_cut.txt", "w", encoding="utf-8")
fa = open("answer2_15w_cut.txt", "w", encoding="utf-8")

english = "〒_=*~`?!.;,⊙o！？｡。＂＃＄％＆＇（）＊＋，－／：＜＝＞｀～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧abcdefghijklmnopqrstuvwxyz0123456789"

def fenci(str):
	output = []
	buffers = ""
	for s in str:
		if s in english or s in english.upper():
			buffers += s
		else:
			if buffers:
				output.append(buffers)
			buffers = ""
			output.append(s)
	if buffers:
		output.append(buffers)
	outstr = " ".join(output)
	return outstr

max_n = 909328/2
# print(np.arange(0, max_n).tolist())
the_choose = random.sample(np.arange(0, max_n).tolist(), 200000)
# print(the_choose)
the_choose.sort()

n = 0
while True:
	lineq = f.readline()
	linea = f.readline()
	if (not lineq) or (not linea):
		break
	n += 1
	if n % 10000 == 0:
		print(n)
	if n in the_choose:
	# if True:
		lineq = cc.convert(lineq)
		linea = cc.convert(linea)
		lineq = fenci(lineq)
		linea = fenci(linea)
		the_choose.remove(n)
		fq.writelines(lineq)
		fa.writelines(linea)
	