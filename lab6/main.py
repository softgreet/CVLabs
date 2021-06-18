import cv2
from items import MessageItem
import os

class Tracker:
    '''
 追踪者模块,用于追踪指定目标
 '''

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        '''
  初始化追踪器种类
  '''
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'GOTURN', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            elif tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            elif tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            elif tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            elif tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            elif tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            elif tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()
            else: # tracker_type == "MOSSE"
                self.tracker = cv2.TrackerMOSSE_create()


    def initWorking(self, frame, box):
        '''
  追踪器工作初始化
  frame:初始化追踪画面
  box:追踪的区域
  '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
  开启追踪
  '''
        message = None
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                                      (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
        return MessageItem(frame, message)


if __name__ == '__main__':
    border = 0
    root = './test_public'
    num=0
    for file in os.listdir(root):
        d = os.path.join(root, file)
        if os.path.isdir(d):
            num+=1
            print(num)
            with open(d+'.txt', 'w') as f:
                tracker = Tracker(tracker_type="CSRT")

                imgs = sorted([img for img in os.listdir(d) if img.endswith('.jpg')], reverse=False, key=lambda x:int(x[:-4]))
                imgs = [os.path.join(d, img) for img in imgs]
                with open(os.path.join(d, 'groundtruth.txt'), 'r') as groundtruth:
                    s = groundtruth.readline()
                    f.write(s)
                nums = [float(i) for i in s.split(',')]
                x = [nums[i] for i in range(0,7,2)]
                y = [nums[i] for i in range(1,8,2)]
                bbox = [int(min(x))-border,int(min(y))-border,int(max(x)-min(x))+border*2,int(max(y)-min(y))+border*2]
                frame = cv2.imread(imgs[0])
                tracker.initWorking(frame, bbox)
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
                # cv2.imshow("track", frame)
                # k = cv2.waitKey(1) & 0xff
                # cv2.destroyAllWindows()
                # bbox = cv2.selectROI(frame, False)
                last = None
                cur = None
                # last_index = 0
                # interrupt = False
                for i in range(1, len(imgs)):
                    img = imgs[i]
                    frame = cv2.imread(img)
                    item = tracker.track(frame)
                    if item.getMessage():
                        nums = item.getMessage()["coord"][0]
                        cur = [nums[0][0], nums[0][1], nums[1][0], nums[0][1], nums[0][0], nums[1][1], nums[1][0],nums[1][1]]
                        # if interrupt:
                        #     steps = [(cur[j]-last[j])/(i-last_index) for j in range(8)]
                        #     for j in range(1, i-last_index):
                        #         f.write('{},{},{},{},{},{},{},{}\n'.format(*[last[k]+steps[k]*j for k in range(8)]))
                        # f.write('{},{},{},{},{},{},{},{}\n'.format(*cur))
                        last = cur
                        # last_index = i
                        # interrupt = False
                        # cv2.imshow("track", item.getFrame())
                        # k = cv2.waitKey(1) & 0xff
                        # if k == 27:
                        #     break
                    # else:
                    #     interrupt = True
                    f.write('{},{},{},{},{},{},{},{}\n'.format(*last))
                # if last_index!=len(imgs)-1:
                #     for j in range(last_index+1, len(imgs)):
                #         f.write('{},{},{},{},{},{},{},{}\n'.format(*last))
                # cv2.destroyAllWindows()
