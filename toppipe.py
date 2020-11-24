"""
Compute topological features for MNIST images
"""

import time
import os.path
import numpy as np
import gtda.images as img

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from gtda.diagrams import Amplitude
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import Scaler
from gtda.homology import CubicalPersistence
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from mnistextr import *

# Some global constants
num_jobs = -1 # Number of jobs for our computations
rand_state = 2020 # A unified random state for the entire execution

def elapsed_time(previous):
    """
    A simple timer that returns the time in seconds since previous[0] and
    updates previous[0] with the current time.
    """
    old, previous[0] = previous[0], time.perf_counter()
    return previous[0] - old

def extract_top_features(X, filtrations, vectorizations):
    """
    Extracts topological features from a MNIST-like dataset. 
    
    For each specified filtration and vectorization, features are extracted
    according to the pipeline:
    Filtration -> Persistence diagram -> Rescaling -> Vectorization.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 28, 28)
        A collection of greyscale images.
        
    filtrations : list of tuples (string, filtration)
        A list of filtrations.
        Assumptions: 1) The first filtration is 'Voxel', the second is
                        'Binary', and for both of them the pipeline is
                        to be run on the original greyscale images. For all
                        subsequent filtrations, the pipeline is to be run on
                        binarized images.
                     2) For all filtrations except 'Vietoris-Rips', the
                        corresponding diagram is the cubical persistence
                        diagram. For 'Vietoris-Rips', i's the Vietoris-Rips
                        persistence diagram.
                    
    vectorizations : list of tuples (string, vectorization)
        A list of vectorizations.
        
    Returns
    -------
    X_f : ndarray of shape (n_samples, n_features)
        Topological features for all images in X
        
    """
    # Put all vectorizations together for convenience
    vect_union = FeatureUnion(vectorizations, n_jobs=num_jobs)
    
    X_bin = img.Binarizer(threshold=0.4, n_jobs=num_jobs).fit_transform(X)
    
    X_f = np.array([]).reshape(X.shape[0], 0)
    current_time = [time.perf_counter()]
    for filt in filtrations:
        filt_features = make_pipeline(\
            filt[1],\
            VietorisRipsPersistence(n_jobs=num_jobs) if filt[0] == 'Vietoris-Rips' else CubicalPersistence(n_jobs=num_jobs),\
            Scaler(n_jobs=num_jobs),\
            vect_union).fit_transform(X)
        X_f = np.hstack((X_f, filt_features)) 
        print("{} complete: {} seconds".format(filt[0], elapsed_time(current_time)))
        if filt[0] == 'Binary': X = X_bin # From now on, we only work with binarized images
            
    return X_f
                       
# Converting MNIST into ndarray format
current_time = [time.perf_counter()]
with open('train-images-idx3-ubyte', mode='rb') as train_imag_file:
    train_imag_full = idx_to_ndarray(train_imag_file)
    print("Training images:\n\t\t\tShape {}\n\t\t\tTotal size {}\n\t\t\tTime {} seconds\n".format(\
            train_imag_full.shape, train_imag_full.size, elapsed_time(current_time)))
    
with open('train-labels-idx1-ubyte', mode='rb') as train_labels_file:
    train_labels_full = idx_to_ndarray(train_labels_file)
    print("Training labels:\n\t\t\tShape {}\n\t\t\tTotal size {}\n\t\t\tTime {} seconds\n".format(\
            train_labels_full.shape, train_labels_full.size, elapsed_time(current_time)))
        
with open('t10k-images-idx3-ubyte', mode='rb') as test_imag_file:
    test_imag = idx_to_ndarray(test_imag_file)
    print("Test images:\n\t\t\tShape {}\n\t\t\tTotal size {}\n\t\t\tTime {} seconds\n".format(\
            test_imag.shape, test_imag.size, elapsed_time(current_time)))
        
with open('t10k-labels-idx1-ubyte', mode='rb') as test_labels_file:
    test_labels = idx_to_ndarray(test_labels_file)
    print("Test labels:\n\t\t\tShape {}\n\t\t\tTotal size {}\n\t\t\tTime {} seconds\n".format(\
            test_labels.shape, test_labels.size, elapsed_time(current_time)))
    
# Randomly choosing 40000 images from the full training set. Since we do not need a validation set,
# the other images are simply discarded.
train_imag, _ , train_labels, _ = train_test_split(train_imag_full, train_labels_full,
                                                   train_size=40000, random_state=rand_state) 
   
# Initializing our filtrations and vectorizations
filtrations = [('Voxel', 'passthrough'),\
               ('Binary', img.Binarizer(threshold=0.4, n_jobs=num_jobs)),\
               ('Height NW', img.HeightFiltration(direction=np.array((-1,-1)), n_jobs=num_jobs)),\
               ('Height N', img.HeightFiltration(direction=np.array((0,-1)), n_jobs=num_jobs)),\
               ('Height NE', img.HeightFiltration(direction=np.array((1,-1)), n_jobs=num_jobs)),\
               ('Height E', img.HeightFiltration(direction=np.array((1,0)), n_jobs=num_jobs)),\
               ('Height SE', img.HeightFiltration(direction=np.array((1,1)), n_jobs=num_jobs)),\
               ('Height S', img.HeightFiltration(direction=np.array((0,1)), n_jobs=num_jobs)),\
               ('Height SW', img.HeightFiltration(direction=np.array((-1,1)), n_jobs=num_jobs)),\
               ('Height W', img.HeightFiltration(direction=np.array((-1,0)), n_jobs=num_jobs)),\
               ('Radial UL', img.RadialFiltration(center=np.array((6,6)), n_jobs=num_jobs)),\
               ('Radial UC', img.RadialFiltration(center=np.array((13,6)), n_jobs=num_jobs)),\
               ('Radial UR', img.RadialFiltration(center=np.array((20,6)), n_jobs=num_jobs)),\
               ('Radial CL', img.RadialFiltration(center=np.array((6,13)), n_jobs=num_jobs)),\
               ('Radial C', img.RadialFiltration(center=np.array((13,13)), n_jobs=num_jobs)),\
               ('Radial CR', img.RadialFiltration(center=np.array((20,13)), n_jobs=num_jobs)),\
               ('Radial DL', img.RadialFiltration(center=np.array((6,20)), n_jobs=num_jobs)),\
               ('Radial DC', img.RadialFiltration(center=np.array((13,20)), n_jobs=num_jobs)),\
               ('Radial DR', img.RadialFiltration(center=np.array((20,20)), n_jobs=num_jobs)),\
               ('Density 2', img.DensityFiltration(radius=2, n_jobs=num_jobs)),\
               ('Density 4', img.DensityFiltration(radius=4, n_jobs=num_jobs)),\
               ('Density 6', img.DensityFiltration(radius=6, n_jobs=num_jobs)),\
               ('Dilation', img.DilationFiltration(n_jobs=num_jobs)),\
               ('Erosion', img.ErosionFiltration(n_jobs=num_jobs)),\
               ('Signed distance', img.SignedDistanceFiltration(n_jobs=num_jobs)),\
               ('Vietoris-Rips', img.ImageToPointCloud(n_jobs=num_jobs))]
vectorizations = [('Bottleneck', Amplitude(metric='bottleneck', order=None, n_jobs=num_jobs)),\
                  ('Wasserstein L1', Amplitude(metric='wasserstein', metric_params={'p': 1}, order=None, n_jobs=num_jobs)),\
                  ('Wasserstein L2', Amplitude(metric='wasserstein', metric_params={'p': 2}, order=None, n_jobs=num_jobs)),\
                  ('Betti L1', Amplitude(metric='betti', metric_params={'p': 1}, order=None, n_jobs=num_jobs)),\
                  ('Betti L2', Amplitude(metric='betti', metric_params={'p': 2}, order=None, n_jobs=num_jobs)),\
                  ('Landscape L1 k=1', Amplitude(metric='landscape', metric_params={'p': 1, 'n_layers':1}, order=None, n_jobs=num_jobs)),\
                  ('Landscape L1 k=2', Amplitude(metric='landscape', metric_params={'p': 1, 'n_layers':2}, order=None, n_jobs=num_jobs)),\
                  ('Landscape L2 k=1', Amplitude(metric='landscape', metric_params={'p': 2, 'n_layers':1}, order=None, n_jobs=num_jobs)),\
                  ('Landscape L2 k=2', Amplitude(metric='landscape', metric_params={'p': 2, 'n_layers':2}, order=None, n_jobs=num_jobs)),\
                  ('Heat kernel L1 sigma=10', Amplitude(metric='heat', metric_params={'p': 1, 'sigma':10}, order=None, n_jobs=num_jobs)),\
                  ('Heat kernel L1 sigma=15', Amplitude(metric='heat', metric_params={'p': 1, 'sigma':15}, order=None, n_jobs=num_jobs)),\
                  ('Heat kernel L2 sigma=10', Amplitude(metric='heat', metric_params={'p': 2, 'sigma':10}, order=None, n_jobs=num_jobs)),\
                  ('Heat kernel L2 sigma=15', Amplitude(metric='heat', metric_params={'p': 2, 'sigma':15}, order=None, n_jobs=num_jobs)),\
                  ('Persistence entropy', PersistenceEntropy(n_jobs=num_jobs))]


if os.path.exists("train_labels.csv"):
    print("train_labels.csv already exists, skipping labels for training images")
else:
    np.savetxt("train_labels.csv", train_labels, delimiter=",")
    print("Labels for training images written into train_labels.csv")


if os.path.exists("train_features.csv"):
    print("train_features.csv already exists, skipping features for training images")
else:    
    print("FEATURE EXTRACTION FOR TRAINING IMAGES STARTS")
    current_time = [time.perf_counter()]
    results_train = extract_top_features(train_imag, filtrations, vectorizations)
    print("FEATURE EXTRACTION FOR TRAINING IMAGES COMPLETE: {} seconds".format(elapsed_time(current_time)))
    np.savetxt("train_features.csv", results_train, delimiter=",")
    print("Features for training images written into train_features.csv")
    
if os.path.exists("test_labels.csv"):
    print("test_labels.csv already exists, skipping labels for test images")
else:
    np.savetxt("test_labels.csv", test_labels, delimiter=",")
    print("Labels for test images written into test_labels.csv")

if os.path.exists("test_features.csv"):
    print("test_features.csv already exists, skipping features for test images")
else:
    print("FEATURE EXTRACTION FOR TEST IMAGES STARTS")
    current_time = [time.perf_counter()]
    results_test = extract_top_features(test_imag, filtrations, vectorizations)
    print("FEATURE EXTRACTION FOR TEST IMAGES COMPLETE: {} seconds".format(elapsed_time(current_time)))
    np.savetxt("test_features.csv", results_test, delimiter=",")
    print("Features for test images written into test_features.csv")
