import cv2
import os
import numpy as np
from eval import *
# 修改了eval，将acc的计算封装成了个函数
gallery = './split/gallery1'
recognizer = cv2.face.LBPHFaceRecognizer_create()
frontal_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profile_detector = cv2.CascadeClassifier("haarcascade_profileface.xml")
min_size = 345
scale_factor, min_neighbors = 1.05, 13

# max_acc, best_min_size, best_scale_factor, best_min_neighbors = 0, 0, 0, 0

def train():
    faces, labs = [], []
    nums = 0
    for f in os.listdir(gallery):
        lab = int(f[:-4])
        img = cv2.imread(os.path.join(gallery, f),cv2.IMREAD_GRAYSCALE)
        w, h = img.shape
        if w < min_size:
            img = cv2.resize(img, dsize=(min_size, h * min_size // w))
        areas = frontal_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        if len(areas) != 0:
            x,y,w,h = areas[0]
            # identified_sizes.append(img.shape)
            faces.append(img[y:y + h, x:x + w])
            labs.append(lab)
        else:
            areas = profile_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
            if len(areas) != 0:
                x, y, w, h = areas[0]
                # identified_sizes.append(img.shape)
                faces.append(img[y:y + h, x:x + w])
                labs.append(lab)
            else:
                img = cv2.flip(img, 1)
                areas = profile_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                if len(areas) != 0:
                    x, y, w, h = areas[0]
                    # identified_sizes.append(img.shape)
                    faces.append(img[y:y + h, x:x + w])
                    labs.append(lab)
                else:
                    nums += 1
                    # not_identified_sizes.append(img.shape)
                    # print(lab, end=' ')
            # cv2.imshow('result image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # print('x,y,w,h: {} {} {} {}'.format(x, y, w, h))
        # cv2.rectangle(img, (x, y, x + w, y + h), color=(0, 255, 0), thickness=2)  # color=BGR
        # cv2.imshow('result image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    recognizer.train(faces, np.array(labs))
    recognizer.write('trainer.yml')
    # print()
    # print(nums)
    # for i in identified_sizes:
    #     print(i, end='\t')
    # print()
    # for i in not_identified_sizes:
    #     print(i, end='\t')
    # print()


def predict(root, filename):
    nums = 0
    with open(filename, mode='w') as file:
        for f in os.listdir(root):
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            w, h = img.shape
            if w < min_size:
                img = cv2.resize(img, dsize=(min_size, h * min_size // w))
            areas = frontal_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
            if len(areas) != 0:
                x,y,w,h = areas[0]
                # identified_sizes.append(img.shape)
                id, confidence = recognizer.predict(img[y:y+h,x:x+w])
                file.write('{} {}\n'.format(f, id))
                print('{} {} {}'.format(f, id, confidence))
            else:
                areas = profile_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                if len(areas) != 0:
                    x, y, w, h = areas[0]
                    # identified_sizes.append(img.shape)
                    id, confidence = recognizer.predict(img[y:y + h, x:x + w])
                    file.write('{} {}\n'.format(f, id))
                    print('{} {} {}'.format(f, id, confidence))
                else:
                    img = cv2.flip(img, 1)
                    areas = profile_detector.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                    if len(areas) != 0:
                        x, y, w, h = areas[0]
                        # identified_sizes.append(img.shape)
                        id, confidence = recognizer.predict(img[y:y + h, x:x + w])
                        file.write('{} {}\n'.format(f, id))
                        print('{} {} {}'.format(f, id, confidence))
                    else:
                        nums += 1
    print(nums)


if __name__ == '__main__':
    if os.path.exists('trainer.yml'):
        recognizer.read('trainer.yml')
    else:
        train()
    # os.remove('val_result.txt')
    # log = open('log.txt', 'a')
    root = './split/sub1'
    filename = 'val_result.txt'
    predict(root, filename)
    acc = getACC()
    print(acc)
    # for i in range(360, 425, 5):
    #     min_size = i
    #     for j in range(5, 10, 5):
    #         scale_factor = 1 + j/100
    #         for k in range(6, 17):
    #             min_neighbors = k
    #             s = 'min_size: {}, scale_factor: {}, min_neighbors: {},'.format(min_size, scale_factor, min_neighbors)
    #             print(s, end=' ')
    #             log.write(s)
    #             os.remove('trainer.yml')
    #             train()
    #             os.remove('val_result.txt')
    #             predict(root, filename)
    #             acc = getACC()
    #             log.write(' ACC: {:.4f}\n'.format(acc))
    #             if acc > max_acc:
    #                 max_acc, best_min_size, best_min_neighbors, best_scale_factor = acc, min_size, min_neighbors, scale_factor
    # print('max_acc: {:.4f}, best_min_size: {}, best_min_neighbors: {}, best_scale_factor: {}'.format(max_acc, best_min_size, best_min_neighbors, best_scale_factor))
    # log.close()

# min_size: 345, scale_factor: 1.05, min_neighbors: 14, ACC: 0.1467
