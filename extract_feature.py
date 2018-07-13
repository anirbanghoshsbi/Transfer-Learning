#usage : python extract_feature.py --dataset ../datasets/{class_label}/example.jpg --output ../datasets/{class_label}/hdf5/features.hdf5

#Feature extraction from a pre trained network 'vgg' using imagenet dataset

#import necessary packages

from keras.application import VGG16             # load the model from whose weights we need
from keras.preprocessing import imagenet_utils        # special helper function for imagenet dataset
from keras.preprocessing.image import img_to_array     # makes the image compatible for keras
from keras.preprocessing.image import load_img        # for loading images from disk
from sklearn.preprocessing import LabelBinarizer       # converts categorical data to integer labels
from imutils import paths        # handy utility for parsing images in a folder
from pyimagesearch.io import HDF5DatasetWriter      # utility to write the model weights as HDF5.
import numpy as np
import argparse        # to construct a argument parser
import random       # to shuffle the data
import os          # to interface with the operating system

# Construct the argument parser

ap=argparse.ArgumentParser()
ap.add_argument('-d','--datatset' , required = True , help = 'Path to the dataset')
ap.add_argument('-o','--output', required =True , help = 'Path where the output HDF5 file would be stored')
ap.add_argument('-b', '--bs', type =int , default= 32, help = 'batch size of images to be passed to the network')
ap.add_argument('-s','--buffer-size', type =int , default =1000, help = 'size of the buffer')

args = vars(ap.parse_args())

#store batch size in a variable

bs = args["batch-size"]

# load the images 
imagepaths = list(paths.list_images(args['dataset']))
random.shuffle(imagePaths)        # let me shuffle the images

# extract the labels : here the consideration is that the images are stored as dataset_name/{class_name}/example.jpg

labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = labelEncoder()
labels = le.fit_transform(labels)

# load the vgg network with imagenet weights...
print('[INFO] loading the network with iamgenet weights...')
model = VGG16(weights = 'imagenet' , include_top=False)

#start the HDF5 dataset writer

dataset = HDF5DatasetWriter(len(imagePaths) , 512 * 7 * 7), args['output'] , datakKey='features',bufSize=['buffer_size'])
dataset.storeClassLabel(le.classes_)

# loop over the images in batches

for i in np.arange(0 , len(imagePaths) , bs):
    batchPaths=imagePaths[i:i+bs]
    batchLabels= labels[i:i+bs]
    batchImages=[]
    for (j , imagepath) in enumerate(batchPaths):
        image = load_img(imagePath , target_size =(224 ,224))
        image = img_to_array(image)
        
        # preprocess the image for imagenet by expanding the 
        # dimensions and substracting the mean RGB pixel
        image = np.expand_dims(image , axis=0)
        image = imagenet_utils.preprocess_input(image)
        #append
        batchImages.append(image)
        
 # Do the forward pass 
 batchImages = np.vstack(batchImages)
 features= model.predict(batchImages , batch_size =bs)
        
 # now flatten the feature vector as we need it to be fed to a logistic regressor later
 features = features.reshape((features.shape[0] , 512*7*7))
        
 # append the dataset
 dataset.add(features , batchlabels)
 dataset.close()       
      



