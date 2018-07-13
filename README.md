{ Based on Implementation in dl4cv book for CV by Dr. Adrian Rosebrock}
# Transfer-Learning
How to use Pre-trained Network to learn patterns from data it was not trained on originally.

In transfer learning we try to use an existing pre-trained network (classifier) as a existing starting point and use it to for a new classification task. 

For example we can take a pretrained classifier like the vgg network trained on imagenet and use it to classify data belonging to Flower-17 or CALTECH-101.

The process by which we do this is called _transfer learning_ here we treat network as arbritary feature extractor. When considering the network as feature extractor we cut the network at a specific point normally at an arbritary point. The CNN at this point is not capable of doing classification , however , given the feature vector that we have now , we can train a classifier like logistic regressor to classify images belonging to say CALTECH -101.

VGG Network :-

Input(224 * 224 * 3) --> (Conv * 2)+Pool [112 * 112 * 128#]--> (Conv * 2) + Pool [56 * 56 * 128]--> (Conv * 3) +Pool [28 * 28 * 128]--> (Conv * 3) +Pool[ 14 * 14 * 128] --> (Conv * 3) +Pool [7 * 7 * 512@]-->FC * 3 + Softmax [1 * 1 * 1000 ] ===>Output labels _1000 classes_
       
#number of filters used 128.
@number of filters used 512.

Transfer learning :-
In this specific problem we remove the FC layer ,so our output from the network is a feature vector of size 512 * 7 * 7 = 25,088.

Input(224 * 224 * 3) --> (Conv * 2)+Pool [112 * 112 * 128#]--> (Conv * 2) + Pool [56 * 56 * 128]--> (Conv * 3) +Pool [28 * 28 * 128]--> (Conv * 3) +Pool[ 14 * 14 * 128] --> (Conv * 3) +Pool [7 * 7 * 512@]-->  _{FC * 3 + Softmax ===>Output labels}_

Every thing within _{}_ gets removed , we are left with a feature vector of 25,088. Later we add a logistic regression module to classify images belonging to say Flower -17. _The important thing to note here is that CNN was not trained on this data._

```
`keras code for cutting the network at specific point`
model = VGG16(weights = 'imagenet', include_top=False)
```

# Extracting the Features 
Step 1 :
Supposing we are going to use the Flower 17 dataset . So we forward propagate the flowers dataset ( just as we would have done in case of drawing inference from the images.) through the vgg network, The output that we obtain from the network is a dataset of size 1360 images having 25088 features each.We store the data in the form of HDF5 file having three headers : features , label_names , labels

where feature.shape = 25,088
label.shape = 1360 (no if pics of flowers)
label_names.shape = 17 there are 17 classes of flower.
Step 2 :
We now train a logistic regressor on this ,feature vector as this is now a basically a 17-class classification problem.

part a ) load the HDF5 file from the database , 
b) split theb dataset into 75 - 25 for train and test
c) Do a grid search to find the optimal hyperparameters to tune for the logistic regressor.
d)train the network and 
e) make predictions and
f) print the result.

Results : training a Logistic Regression Model on Flower -17 on a pre-trained weights of imagenet dataset of vgg network gives 93% classification accuracy.

_Therefore the Network is able to perform transfer learning , encoding the discriminative features on a output activations that we can use to train our own custom image classifier._

# Advantage of Transfer learning :-
if feature extraction with reasonable accuracy is obtained using transfer learning it can save lots of time , effort and compute power.
