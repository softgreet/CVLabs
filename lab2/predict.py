from PIL import Image

from yolo import YOLO

import os

import json

yolo = YOLO()


path = "./VOCdevkit/VOC2007/val/"
dirs = os.listdir(path)
results={}
f = open(os.path.join("./",'my_val_epoch10_05_01.json'), 'w')

for file in dirs:
    try:
        image = Image.open(path+file)
    except:
        print('Open Error! Try again!')
        continue
    else:
        result = dict()
        result['height']=int(image.height)
        result['width']=int(image.width)
        result['depth']=3
    
        print(image)
        r_image, classify = yolo.detect_image(image)

        objects={}
        for i in range(0,len(classify)):
            obj = dict()
            obj['category']=str(classify[i][4])
            obj['bbox']=classify[i][0:4]
            objects[str(i)]=obj
        result['objects']=objects
        results[str(file)]=result

        print(file)


str=json.dumps(results)
f.write(str)
f.close()
