import cv2
import dlib
import os

if __name__ == '__main__':
    border = 0
    root = r'./trainval'
    num=0
    for file in os.listdir(root):
        d = os.path.join(root, file)
        if os.path.isdir(d):
            num+=1
            print(num)
            with open(d+'.txt', 'w') as f:
                tracker = dlib.correlation_tracker()
                imgs = sorted([img for img in os.listdir(d) if img.endswith('.jpg')], reverse=False, key=lambda x:int(x[:-4]))
                imgs = [os.path.join(d, img) for img in imgs]
                with open(os.path.join(d, 'groundtruth.txt'), 'r') as groundtruth:
                    s = groundtruth.readline()
                    f.write(s)
                nums = [float(i) for i in s.split(',')]
                x = [nums[i] for i in range(0,7,2)]
                y = [nums[i] for i in range(1,8,2)]
                bbox = dlib.rectangle(int(min(x)),int(min(y)),int(max(x)),int(max(y)))
                frame = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
                # temp_img = frame.copy()
                # cv2.rectangle(temp_img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
                # cv2.imshow('image',temp_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                tracker.start_track(frame, bbox)
                last = None
                cur = None
                last_index = 0
                interrupt = False
                for i in range(1, len(imgs)):
                    img = imgs[i]
                    frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                    confidence = tracker.update(frame)
                    box_predict = tracker.get_position()
                    left = box_predict.left()
                    top = box_predict.top()
                    right = box_predict.right()
                    bottom = box_predict.bottom()
                    f.write('{},{},{},{},{},{},{},{}\n'.format(left,top,right,top,left,bottom,right,bottom))