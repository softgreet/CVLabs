import shutil
import os

if __name__ == '__main__':
    root = r'.\trainval'
    for file in os.listdir(root):
        d = os.path.join(root, file)
        if os.path.isdir(d):
            shutil.copyfile(os.path.join(d, 'groundtruth.txt'), os.path.join('test_annotation_files', file+'.txt'))