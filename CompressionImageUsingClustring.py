import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# The below two are visualization libraires
import matplotlib.pyplot as plt
import seaborn as sns 

# for calculating interval
from time import time

plt.rcParams['figure.figsize'] = 10,8 # définition de la taille de figure par défaut pour le noyau

# for clustering image pixels
from sklearn.cluster import KMeans 

"""
https://medium.com/@agarwalvibhor84/image-compression-using-k-means-clustering-8c0ec055103f
"""
"""
https://www.kaggle.com/psvishnu/image-compression-using-clustering/comments#675109
"""


#pour regrouper les pixels d'image
from skimage import io
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTzLzdD4MoAPP3I_JjrxndAqKg1zXtEnaODwsHbH6il9BL3Qt61'
img_original = io.imread(url)
plt.axis('off')
plt.imshow(img_original)
plt.title('Our buddy for the experiment !')
plt.show()
#matrice de img de l'img
print(img_original)
#taille de img
print("taille de img = ",img_original.shape[0], "*" , img_original.shape[1])
#☺mode img
"""import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('C:/Users/kais2/OneDrive/Bureau/chien.jpg',cv.IMREAD_GRAYSCALE)
roihist = cv.calcHist([img],[0], None, [256], [ 0, 256] )
xs=np.linspace(0,255,256)
plt.subplot(2,1,1)
plt.plot(xs,roihist,color='k')
img = cv.imread('C:/Users/kais2/OneDrive/Bureau/chien.jpg',cv.IMREAD_COLOR)
color = ('b','g','r')
plt.subplot(2,1,2)
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
"""

# Unit normalizing 
img = np.array(img_original,dtype=float) / 255

# Save the dimensions, we will be need them later
w, h, d = original_shape = img.shape
print('Original Shape'.center(20,'='))
print(img.shape)

# image_array size - w*h , d
image_array = img.reshape(-1,d)
print('ReShaped'.center(20,'='))
print(image_array.shape)
n_colours = [64,32]

# 64 colour image
t0 = time()
kmeans64 = KMeans(n_clusters = n_colours[0],random_state=42,verbose=2,n_jobs=-1).fit(image_array)

print(f'Completed 64 clusters in {round(time()-t0,2)} seconds.')

# 32 colour image
t0 = time()
kmeans32 = KMeans(n_clusters = n_colours[1],random_state=42,verbose=2,n_jobs=-1)
kmeans32.fit(image_array)

print(f'Completed 32 clusters in {round(time()-t0,2)} seconds.')
print(f'Within cluster sum of square error for {n_colours[0]} clusters = {round(kmeans64.inertia_,2)}')
print(f'Within cluster sum of square error for {n_colours[1]} clusters = {round(kmeans32.inertia_,2)}')
# checking the compressed values
compressed = pd.DataFrame(image_array,columns=['Red','Green','Blue'])
compressed['labels'] = kmeans64.labels_
compressed

labels64 = kmeans64.labels_
labels32 = kmeans32.labels_
# Recreate image
def recreate_image(centroids, labels, w, h):
    # centroids variable are calculated from the flattened image
    # centroids: w*h, d 
    # so each row depicts the values per depth
    d = centroids.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            # filling values in new image with centroid values
            image[i][j] = centroids[labels[label_idx]]
            label_idx += 1
    return image
plt.figure(figsize=(20,10))
plt.subplot(132)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

plt.subplot(131)
plt.axis('off')
plt.title('Compressed image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans64.cluster_centers_, labels64, w, h))

plt.subplot(133)
plt.axis('off')
plt.title('Compressed image (32 colors, K-Means)')
plt.imshow(recreate_image(kmeans32.cluster_centers_, labels32, w, h))

plt.show()












