# -*- coding: utf-8 -*-
# @Author  : LG
import hashlib
import os

def GetFileMd5(filename):
    if not os.path.isfile(filename):
        return
    myHash = hashlib.md5()
    f = open(filename,'rb')
    while True:
        b = f.read(8096)
        if not b :
            break
        myHash.update(b)
    f.close()
    return myHash.hexdigest()

