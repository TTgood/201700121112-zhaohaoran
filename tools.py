# -*- coding: utf-8 -*-
# @Author: zhr1030635594
# @Date:   2019-12-12 02:38:36
# @Last Modified by:   zhr1030635594
# @Last Modified time: 2019-12-12 02:38:57
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