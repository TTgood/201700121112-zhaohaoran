# -*- coding: utf-8 -*-
# @Author: zhr1030635594
# @Date:   2019-12-12 15:15:40
# @Last Modified by:   zhr1030635594
# @Last Modified time: 2019-12-12 16:47:49
import json

places = open("places.txt", "r")
n = 0
n1, n2, n3 = 0, 0, 0
place_list = []
while True:
	n += 1
	line = places.readline()
	if not line:
		print(n)
		break
	
	line = str(line)
	place = line.split()[-1]
	if place[-1] in ["市", "区", "乡", "镇", "省", "村", "旗", "县", "盟", "州"]:
		pure_place = place[:-1]
		n1 += 1
	elif place[-2:] in ["街道", "地区", "市区"]:
		pure_place = place[:-2]
		n2 += 1
	elif place[-3:] in ["自治区", "自治县", "自治乡", "管理区", 
				"管理处", "自治州"]:
		pure_place = place[:-3]
		n3 += 1
	else:
		pure_place = place
	place_list.append(pure_place)
print(len(place_list))
print(place_list[:5])
# json_places = json.dumps(place_list)

with open("places_json.txt", "w") as f2:
	f2.writelines(json.dumps(place_list, ensure_ascii=False))
# print(n1, n2, n3)
