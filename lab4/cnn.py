import PIL
import face_recognition
import os

from PIL import Image

from read_img import endwith
import numpy as np
import random
import cv2 as cv


def _encode_img(img, num_jitters=2, treshold=300):
    """Resize image and return encodings.

    num_jitters: The higher is the more accurate but proportionaly slower.
    10 is 10 times slower than 1. The default is 1.

    treshold: Maximum dimension in image size. Larger images will be downscaled to the treshold.
    The defaul value is 1000.
    """
    img_enc = []

    # img_load = cv.imread(img)
    # # face_recognition works with RGB
    # img_load = cv.cvtColor(img_load, cv.COLOR_BGR2RGB)
    # x, y, h = img_load.shape
    # if max(x, y) > treshold:
    #     scale = float(1 / (max(x, y) / treshold))
    #     img_load = cv.resize(img_load, None, fx=scale, fy=scale)

    im = PIL.Image.open(img)
    x, y = im.size
    im = im.resize((300, 300), Image.ANTIALIAS)
    img = im.convert(im.mode)
    img_load = np.array(img)
    try:
        img_enc = face_recognition.face_encodings(img_load)[0]
    except IndexError:
        face_locations = face_recognition.face_locations(img_load, model='cnn')
        if len(face_locations) == 0:
            print(img, "I wasn't able to locate any faces.")
        else:
            img_enc = face_recognition.face_encodings(img_load, face_locations)[0]
    return img_enc


def get_face(path):    # 获得gallery中的每一张图片的encoding值，作为known_face_encoding
    faces_encoding = []
    img_name = []
    for child_dir in os.listdir(path):
        if endwith(child_dir, 'jpg'):
            child_path = path + '/' + child_dir
            # img = face_recognition.load_image_file(child_path)
            # face_locations = face_recognition.face_locations(img, model='cnn')
            img_face_encoding = _encode_img(child_path)
            img_name.append(child_dir[0:-4])
            faces_encoding.append(img_face_encoding)
            print("Loading " + child_path)
    return faces_encoding, img_name


def get_face_predict(path):    # 获得val中的每一张图片的encoding值，并与known_face_encoding进行compare，找出距离小于tolerance=0.6的
    f = open(os.path.join("./", 'my_val02.txt'), 'w')
    for child_dir in os.listdir(path):
        if endwith(child_dir, 'jpg'):
            child_path = path + '/' + child_dir
            img_face_encoding = _encode_img(child_path)     # 获得faces编码
            if len(img_face_encoding) != 0:    # 检测到人脸
                print("Loaded: " + child_path)
                results = face_recognition.compare_faces(known_face_encoding, img_face_encoding, 0.7)    # 获得tolerance<0.7的图片结果

                if results.count(True)>1 or results.count(True) == 0:    # 有超过一张or没有合适的图片，则调出distance最小的
                    distances = face_recognition.face_distance(known_face_encoding, img_face_encoding).tolist()
                    index=distances.index(min(distances))
                    f.write(child_dir + " " + img_name[index] + '\n')
                    print(child_dir, "The number of True:", results.count(True), "Label:", img_name[index])
                else:    # 否则直接选择result为True的
                    f.write(child_dir + " " + img_name[results.index(True)] + '\n')
                    print(child_dir, "The number of True:", results.count(True), "Label:",img_name[results.index(True)])

            else:    # 未检测到人脸
                f.write(child_dir + " " + str(random.randint(0, 50)) + '\n')
                print(child_dir)
    f.close()
    return "Over!"


known_face_encoding, img_name = get_face("F:/dataset/gallery")
unknown_face_encoding = get_face_predict("F:/dataset/val")
