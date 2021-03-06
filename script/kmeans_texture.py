# -*- coding: utf-8 -*-
"""kmeans_texture.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1quj8IGB9L-Pvhre31Nt9VnGnfx0cZESG
"""


import pickle
import numpy as np
import os

import time





path = ""

with open(os.path.join(path,"label2num.pkl"), "rb") as fp:   # Unpickling
    label2num = pickle.load(fp)
    
with open(os.path.join(path,"num2label.pkl"), "rb") as fp:   # Unpickling
    num2label = pickle.load(fp)
    
with open(os.path.join(path,"train_feature.pkl"), "rb") as fp:   # Unpickling
    train_feature = pickle.load(fp)

with open(os.path.join(path,"train_label.pkl"), "rb") as fp:   # Unpickling
    train_label = pickle.load(fp)

points_np = np.concatenate(train_feature, axis=0)

from sklearn.cluster import MiniBatchKMeans

num_of_classes = 47

for scale in range(1,11):
    number_of_clusters = num_of_classes*scale
    kmeans = MiniBatchKMeans(n_clusters=number_of_clusters,
                            max_iter=1000,
                         batch_size = 1000000
                        )
    start = time.time()

    kmeans = kmeans.fit(points_np)

    end = time.time()
    print(end-start)

    with open(os.path.join(path,"batch_kmeans_model_"+str(number_of_clusters)+".pkl"), "wb") as fp:   #Pickling
        pickle.dump(kmeans, fp)

