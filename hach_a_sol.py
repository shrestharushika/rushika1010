#!/usr/bin/env python
# coding: utf-8




from keras.applications  import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input 

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from google.colab import files
import pickle




vgg_model=vgg16.VGG16('imagenet')


# Remove the last layer in order to get the features




img_feature=Model(inputs=vgg_model.input,outputs=vgg_model.get_layer("fc2").output)
img_feature.summary()




img=files.upload()

original=plt.imread('parle-namkeen-bhujia-sev.jpg')





import cv2
height = 224
width = 224
dim = (width, height)


res = cv2.resize(original, dim, interpolation=cv2.INTER_LINEAR)
original=res





plt.imshow(original)


 




numpy_img=img_to_array(original)





img_batch=np.expand_dims(numpy_img,axis=0)
print('image batch size',img_batch.shape)





#prepare for the vgg model
processed_model=preprocess_input(img_batch.copy())





#get the extracted image

features=img_feature.predict(processed_model)
print("Number of featues",features.size)





#load all the images
files.upload()
imported_img=plt.imread('images.jpg')
res = cv2.resize(imported_img, (224,224), interpolation=cv2.INTER_LINEAR)
imported_img=res





plt.imshow(imported_img)





numpy_img=img_to_array(imported_img)
img_batch=np.expand_dims(numpy_img,axis=0)
# image = np.vstack(imported_img)
print('image batch size',img_batch.shape)




processed_imported_img=preprocess_input(img_batch.copy())





imported_img_feat=img_feature.predict(processed_imported_img )





print('Feature extracted',imported_img_feat.shape)





from sklearn.metrics.pairwise import cosine_similarity





nb_closest_images=0
cos_similarity_score=cosine_similarity(imported_img_feat)
cos_similarity_score





#original image
ori_grayimage= cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
ori_hist=cv2.calcHist([ori_grayimage],[0],None,[256],[0,256])





#test image

img2=cv2.cvtColor(imported_img,cv2.COLOR_BGR2GRAY)
img_hist=cv2.calcHist([img2],[0],None,[256],[0,256])





#calculate eucladians distance between original and imported image

i=0
c1=0
while(i<len(ori_hist) and i<len(img_hist)):
  c1=c1+(ori_hist[i]-img_hist[i])**(2)
  i+=1
c1=c1**(1/2)





c1





print("Original Product")
plt.imshow(original)
plt.show()





print("Similarity Score",cos_similarity_score,"%")
plt.imshow(imported_img)

filename = 'Similarity_model'
pickle.dump(rfcl, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))





