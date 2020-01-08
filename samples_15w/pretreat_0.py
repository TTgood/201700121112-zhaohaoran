# -*- coding: utf-8 -*-
# @Author: zhr1030635594
# @Date:   2019-10-22 22:15:31
# @Last Modified by:   zhr
# @Last Modified time: 2019-11-12 12:18:36
import os
from opencc import OpenCC 

cc = OpenCC("t2s")

file = open("xiaohuangji50w_fenciA.conv", encoding="utf-8")
f1 = open("xhj1_fenci.conv", mode="w", encoding="utf-8")
n = 0
m = 0
in_num = 0
while True:
	
	line = file.readline()
	if line:

		if line[0] == "E":
			continue
		elif line[0] == "M":
			n += 1
			l = line[2:]
			l = cc.convert(l)
			# if n == 1:
			# 	print(l)
			# 	print(l[-1])
			# 	print(len(l))
			f1.writelines(l)
			# n += 1
			# l = line[2:]
			
		else:
			# print("wrong", n)
			m += 1
		# if n % 3 == 0:
		# 	f1.writelines("\n")
	else:
		break
print(m)