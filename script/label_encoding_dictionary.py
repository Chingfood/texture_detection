#!/usr/bin/env python
# coding: utf-8

# %config Completer.use_jedi = False

# In[2]:


import os
import pickle


# In[3]:


train_text = "dtdata/train.txt"
validate_text = "dtdata/val.txt"
test_text = "dtdata/test.txt"


# In[4]:


def label_generating_from_path(file_path):
    label = file_path.strip().split('/')[-1].split('_')[0]
    return label


# In[5]:


"""
Construct all the labels possible and encode it numerically
"""
labels_list = []
with open(train_text) as f:
    for line in f:
        labels_list.append(label_generating_from_path(line))
labels_list_redundancy_removed = list(set(labels_list))
  

labels_list_redundancy_removed.sort()

label2num = {}
num2label = {}
for i, item in enumerate(labels_list_redundancy_removed):
    label2num[item] = i
    num2label[i] = item


# In[8]:


with open("label2num.pkl", "wb") as fp:   #Pickling
    pickle.dump(label2num, fp)
    
with open("num2label.pkl", "wb") as fp:   #Pickling
    pickle.dump(num2label, fp)

