import cv2
import os
import numpy as np
import random
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import math
AUTOTUNE = tf.data.experimental.AUTOTUNE
cv2.setNumThreads(1)
class VimeoDataset(Sequence):
    def __init__(self, dataset_name, path, batch_size=32, shuffle=False, train_root='tri_trainlist.txt',test_root='tri_testlist.txt',mode='full'):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.data_root = path
        self.shuffle=shuffle
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, train_root)
        test_fn = os.path.join(self.data_root, test_root)
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()        

        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.trainlist):
                if i % 100 == 0:
                    tmp.append(value)
            self.trainlist = tmp
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 10 == 0:
                    tmp.append(value)
            self.testlist = tmp

        self.load_data()
        self.on_epoch_end()

    def __len__(self):
        #return len(self.meta_data)
        return math.ceil(len(self.meta_data) /self.batch_size)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.meta_data))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def getimg(self, indices, is_training=False,h=256,w=256):
        img0_list=[]
        img1_list=[]
        gt_list=[]
        for index in indices:
            imgpath = os.path.join(self.image_root, self.meta_data[index]) 
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
            #image = tf.io.read_file(imgpaths[0])
            #image = tf.image.decode_png(image)
            if is_training : 
                img0= cv2.imread(imgpaths[0])
                gt= cv2.imread(imgpaths[1])
                img1= cv2.imread(imgpaths[2])

                ih, iw, _ = img0.shape
                x = np.random.randint(0, ih - h + 1)
                y = np.random.randint(0, iw - w + 1)
                img0 = img0[x:x+h, y:y+w, :]
                img1 = img1[x:x+h, y:y+w, :]
                gt = gt[x:x+h, y:y+w, :]

                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                if random.uniform(0, 1) < 0.5:
                    img1, img0 = img0, img1
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]

                p = random.uniform(0, 1)
                if p < 0.25:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                elif p < 0.5:
                    img0 = cv2.rotate(img0, cv2.ROTATE_180)
                    gt = cv2.rotate(gt, cv2.ROTATE_180)
                    img1 = cv2.rotate(img1, cv2.ROTATE_180)
                elif p < 0.75:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                img0_list.append(img0)
                gt_list.append(gt)
                img1_list.append(img1)
            else :
                img0_list.append(cv2.imread(imgpaths[0]))
                gt_list.append(cv2.imread(imgpaths[1]))
                img1_list.append(cv2.imread(imgpaths[2]))

        #H=512
        #W=512
        #img0 = cv2.resize(img0,(W,H))#1k
        #gt = cv2.resize(gt,(W,H))
        #img1 = cv2.resize(img1,(W,H))
 
        #return np.concatenate((img0_list,gt_list,img1_list),0)
        return np.array(img0_list), np.array(gt_list), np.array(img1_list)
            
    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.meta_data))

        indices = self.indices[low:high]                        
        img0, gt, img1 = self.getimg(indices,is_training='train' in self.dataset_name )  


        return img0,gt, img1

if __name__ == '__main__':
    test_loader=VimeoDataset(dataset_name='test',
               path='/data/dataset/vimeo_dataset/vimeo_triplet',
               shuffle=True)
    for e in range(10):
        for img0,gt,img1 in test_loader:
            pass
