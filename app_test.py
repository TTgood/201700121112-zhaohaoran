# -*- coding: utf-8 -*-
# @Author: zhr1030635594
# @Date:   2019-12-12 15:10:33
# @Last Modified by:   zhr1030635594
# @Last Modified time: 2019-12-12 20:53:46
import requests
import json
import random
import time
from attention_test import chat, make_model

original_name = "小通"
name = "智慧小然"

model = make_model()

# def get_joke():
# 	# t = time.time()
# 	t = 1418745237
# 	url = "http://api.avatardata.cn/Joke/QueryJokeByTime?key=9b94178fa8d54649a350f5dc1d9aa2ad&page=2&rows=20&sort=asc&time={}".format(int(t)-1000000)
# 	print(url)
# 	s = requests.get(url)
# 	response_dic=s.json()
# 	ss = str(response_dic).replace("'", '"')
# 	print(ss)

# get_joke()

# def if_joke(string):
# 	tell = ["给我", "讲", "说"]
# 	flag1 = False
# 	for t in tell:
# 		if t in string:
# 			flag1 = True
# 			break
# 	flag2 = False
# 	if "笑话" in string:
# 		flag1 = True
# 	if flag1 and flag2:
# 		flag = True


# flaces = open()
def weather_api(url):
	s = requests.get(url)
	response_dic=s.json()
	ss = str(response_dic).replace("'", '"')
	text = json.loads(ss)["content"]
	# print(text)
	return text

def whether_weather(string):
	if "天气" not in string:
		return False
	this_place = ""
	f = open("places_json.txt", "r")
	places = json.load(f)
	# print(places[0:5])
	flag = False
	for p in places:
		if p in string:
			if not p:
				continue
			# print(p)
			this_place = p
			flag = True
	# print(this_place)
	# print(flag)
	if flag:
		# print("**")
		# print(this_place)
		my = "http://api.qingyunke.com/"+ "api.php?key=free&appid=0&msg=" + this_place + "天气"
		text = weather_api(my)
		return text
	else:
		print("你是说哪里呀？\n")
		str_2 = input("> ")
		flag = False
		for p in places:
			if p in str_2:
				if not p :
					continue
				this_place = p
				flag = True
		if flag:
			# print("&&")
			my = "http://api.qingyunke.com/"+"api.php?key=free&appid=0&msg=" + this_place + "天气"
			text = weather_api(my)
			return text
		else:
			return [False, str_2]

while True:
	# print("请跟我聊聊吧\n")
	human = input("> ")
	weather_return = whether_weather(human)
	# print(weather_return)
	if weather_return is False:
		out = chat(human, model).replace(original_name, name)
		print(out + "\n")
	if isinstance(weather_return, str):
		print(weather_return+"\n")
	if isinstance(weather_return, list):
		out = chat(weather_return[1], model).replace(original_name, name)
		print(out + "\n")


