# train a Linear classifier for classification purpose

# import necessary packages

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metics import classification_report
import argparse
import pickle
import h5py

ap =argparse.ArgumentParser()
ap.add_argument('-d','--db', required = True , help = 'path to HDF5 database')
ap.add_argument('-m','--model', required = True , help = 'path to output model')
ap.add_argument('-j','--jobs', required = True , type =int ,default=-1, help = '# no of jobs to be done to find the correct hyper parameters')
args=vars(ap.parse_args())


# read the hdf5 file

db = h5py.File(args['db'] , 'r')

i = int(db["labels"].shape[0]*0.75)

# define parameters for GridSearchCV
print('tuning hyperparameters...')
params = {'C' : [0.1,1,10,100,1000]} 
model = GridSearchCV(LogisticRegression() , params , cv =3 , n_jobs =args['jobs'])
print('Choosing the best hyper-parameters : . format(model.best_param_'))

# evaluate the network
preds = model.predict(db["features"][:i])
print('classification Report:' , classification_report(db['labels'][:i] , preds , target_names= db['label_names']))

print("save the model...")

with open(args['model'] ,'wb') as file:
    file.write(pickle.dumps(model.best_estimator_))
db.close()    
