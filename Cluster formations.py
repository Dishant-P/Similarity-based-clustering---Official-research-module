import numpy as np 
import pandas as pd
import scipy 
import pickle
from scipy.spatial import distance as scidist
import sys
sys.path.insert(1, "D:\\Work\\Research\\")
from src.evaluate import distance
from sklearn.cluster import KMeans
import time
import os, shutil

class SMG:
    def __init__(self, dataset):
        file_reader = open(dataset, 'rb')
        self.dataset = pickle.load(file_reader)
        if(type(self.dataset) != pd.core.frame.DataFrame):
            self.dataset = pd.DataFrame(self.dataset)
        self.feature_centroids = {}
        self.simmat = {}
        self.class_index = {}
        self.class_list = []
        self.gen_index()
    
    def gen_feature_centroids(self):
        model = KMeans(n_clusters=1)
        self.class_list = self.dataset['cls'].unique()
        for name in self.class_list:
            if(name not in self.feature_centroids.keys()):
                X = np.array(self.dataset.loc[self.dataset['cls'] == name]['hist'])
                X = np.vstack(X)
                model.fit(X)
                self.feature_centroids[name] = model.cluster_centers_
    
    def gen_similarity_matrix(self):
        model = KMeans(n_clusters=1)
        for index, name in enumerate(self.class_list):
            class_similarity = np.empty((len(self.class_list)))
            for second_index, second_name in enumerate(self.class_list):
                class_similarity[second_index] = distance(self.feature_centroids[name], self.feature_centroids[second_name], d_type="cosine")
            self.simmat[name] = class_similarity
            
    def gen_index(self):
        self.gen_feature_centroids()
        self.gen_similarity_matrix()

start_time = time.time()
test = SMG("D:\\Work\\Research\\Features - Dogs\\resnet-120")
print(round(time.time() - start_time, 1))

dog_breeds = list(test.simmat.keys())

X_train = []
for key, value in test.simmat.items():
    temp_array = list(value)
    X_train.append(temp_array)

from sklearn.cluster import AgglomerativeClustering

#For two clusters
clustering = AgglomerativeClustering(n_clusters=2).fit(X_train)

labels = clustering.labels_

for index, breed in enumerate(dog_breeds):
    temp = breed
    temp = temp.strip("database\\")
    dog_breeds[index] = temp

cluster1, cluster2 = [], []
for index, value in enumerate(labels):
    if(value == 0):
        cluster1.append(dog_breeds[index])
    else:
        cluster2.append(dog_breeds[index])


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

for value in cluster1:
    os.mkdir("D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_2\\c1\\" + str(value))
    src, dst = "D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs\\" + str(value),"D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_2\\c1\\" + str(value) + "\\"
    copytree(src,dst)

for value in cluster2:
    os.mkdir("D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_2\\c2\\" + str(value))
    src, dst = "D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs\\" + str(value),"D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_2\\c2\\" + str(value) + "\\"
    copytree(src,dst)

#For three clusters
clustering = AgglomerativeClustering(n_clusters=3).fit(X_train)

labels = clustering.labels_

cluster1, cluster2, cluster3 = [], [], []
for index, value in enumerate(labels):
    if(value == 0):
        cluster1.append(dog_breeds[index])
    elif(value == 1):
        cluster2.append(dog_breeds[index])
    else:
        cluster3.append(dog_breeds[index])

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
for value in cluster1:
    os.mkdir("D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c1\\" + str(value))
    src, dst = "D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs\\" + str(value),"D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c1\\" + str(value) + "\\"
    copytree(src,dst)

for value in cluster2:
    os.mkdir("D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c2\\" + str(value))
    src, dst = "D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs\\" + str(value),"D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c2\\" + str(value) + "\\"
    copytree(src,dst)
    
for value in cluster3:
    os.mkdir("D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c3\\" + str(value))
    src, dst = "D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs\\" + str(value),"D:\\Work\\Research\\Classification datasets\\Datasets\\Stanford Dogs_3\\c3\\" + str(value) + "\\"
    copytree(src,dst)

