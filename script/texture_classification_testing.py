#!/usr/bin/env python
# coding: utf-8

# %config Completer.use_jedi = False

# In[2]:


from lm import *


# In[3]:


from skimage.color import rgb2gray
from imageio import imread, imsave
import os
import matplotlib.pyplot as plt


# In[4]:


import time
import pickle


# In[5]:


F = makeLMfilters()


# In[6]:


train_text = "dtdata/train.txt"
validate_text = "dtdata/val.txt"
test_text = "dtdata/test.txt"


# In[7]:


def label_generating_from_path(file_path):
    label = file_path.strip().split('/')[-1].split('_')[0]
    return label


# In[8]:


def img_load_gray(file_path):
    file_path = file_path.strip()
    return rgb2gray(imread(file_path))


# In[9]:


def img_filepath_preprocess(path):
    return os.path.join('dtdata', 'images', path)


# In[10]:


def apply_filter(image, filter_matrix, stride=1):
    
    filter_size_x, filter_size_y = filter_matrix.shape[:2]
    
    start_x = int((filter_size_x - 1) / 2)
    start_y = int((filter_size_y - 1) / 2)
    
    output_size_x = int((image.shape[0] - filter_size_x ) / stride + 1)
    output_size_y = int((image.shape[1] - filter_size_y ) / stride + 1)
    
    output = np.zeros([output_size_x,output_size_y,filter_matrix.shape[2]])
    
    for i in range(output_size_x):
        for j in range(output_size_y):
            im_i = start_x + i*stride
            im_j = start_y + j*stride

            patch = np.repeat(image[im_i-start_x : im_i+start_x+1, im_j-start_y : im_j+start_y+1, np.newaxis], F.shape[2], axis=2)
            
            output[i,j] = (patch * filter_matrix).sum(axis = (0,1))
    
    mean_mat = np.tile(output.mean(axis=(0,1))[np.newaxis, np.newaxis,:], output.shape[:2]+(1,))
    std_mat = np.tile(output.std(axis=(0,1))[np.newaxis, np.newaxis,:], output.shape[:2]+(1,))
    normalized_output = (output-mean_mat)/std_mat
    return normalized_output
            


# In[11]:


def output2points(matrix):
    return matrix.reshape(-1,matrix.shape[-1])


# In[12]:


"""
The image and processed feature vector is too big for the RAM, going to crop the image first
"""
def crop_image(img,cropx=150,cropy=150):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


# """
# Construct all the labels possible and encode it numerically
# """
# labels_list = []
# with open(train_text) as f:
#     for line in f:
#         labels_list.append(label_generating_from_path(line))
# labels_list_redundancy_removed = list(set(labels_list))
#   
# 
# labels_list_redundancy_removed.sort()
# 
# label2num = {}
# num2label = {}
# for i, item in enumerate(labels_list_redundancy_removed):
#     label2num[item] = i
#     num2label[i] = item

# In[14]:


with open("label2num.pkl", "rb") as fp:   # Unpickling
    label2num = pickle.load(fp)
    
with open("num2label.pkl", "rb") as fp:   # Unpickling
    num2label = pickle.load(fp)
    


# start = time.time()

# counter = 0

# In[ ]:


points_list = []
label_list = []
with open(test_text) as f:
    for line in f:
#         counter += 1
#         if counter > 10: break
        label_list.append( label2num[label_generating_from_path(line)] )
        
        image = crop_image(img_load_gray(img_filepath_preprocess(line)),200,200)

        points_list.append( (output2points( apply_filter(image, F, stride=2) ) ).astype(np.float32) )
        


# end = time.time()
# print(end-start)

# In[ ]:


with open("test_feature.pkl", "wb") as fp:   #Pickling
    pickle.dump(points_list, fp)

with open("test_label.pkl", "wb") as fp:   #Pickling
    pickle.dump(label_list, fp)


# In[ ]:




