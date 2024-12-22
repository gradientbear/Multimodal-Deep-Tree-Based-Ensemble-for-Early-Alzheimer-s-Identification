from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier
import os
import pickle
import pandas as pd
import numpy as np
from clinica.pipelines.machine_learning import algorithm, validation

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier
)



def get_ML_classifiers():
    """
    Returns a dictionary of classifiers.
    """
    return {
        "Random Forest": RandomForestClassifier(max_depth=2, random_state=0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(),
        "Neural Network": MLPClassifier(
            solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),
            random_state=1, max_iter=1000
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=0),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=0)
    }

def get_ANN_model(input_dim, output_dim):
    """
    Returns an ANN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_CNN_model(input_shape, output_dim):
    """
    Returns a CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_TabNet_model(input_dim, output_dim):
    """
    Returns a TabNet model.
    """
    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5, gamma=1.3,
        n_independent=2, n_shared=2, epsilon=1e-15,
        momentum=0.02, clip_value=2., lambda_sparse=1e-3,
        optimizer_fn=tf.keras.optimizers.Adam,
        optimizer_params=dict(learning_rate=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=tf.keras.optimizers.schedules.ExponentialDecay,
        seed=0, verbose=0
    )
    return model

class TsvRFWf(): #class to run the random forest classification
    def __init__(self, data_tsv, columns, output_dir, n_threads=20, n_iterations=250, test_size=0.2,
                 grid_search_folds=10, balanced=True, n_estimators_range=(100, 200, 400), max_depth_range=[None],
                 min_samples_split_range=[2], max_features_range=('auto', 0.25, 0.5), splits_indices=None,
                 inner_cv=False):

        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._n_estimators_range = n_estimators_range
        self._max_depth_range = max_depth_range
        self._min_samples_split_range = min_samples_split_range
        self._max_features_range = max_features_range
        self._splits_indices = splits_indices
        self._inner_cv = inner_cv
        self._columns=columns

        self._dataframe = data_tsv
        self._validation = None
        self._algorithm = None

    def run(self):
        x = self._dataframe[self._columns].to_numpy()
        unique = list(set(self._dataframe["diagnosis"]))
        y = np.array([unique.index(x) for x in self._dataframe["diagnosis"]])
        
        #apply random forest from algorithm on the parameters given by the user 
        parameters_dict = {
            "balanced": self._balanced,
            "grid_search_folds":  self._grid_search_folds,
            "n_estimators_range": self._n_estimators_range,
            "max_depth_range": self._max_depth_range,
            "min_samples_split_range": self._min_samples_split_range,
            "max_features_range": self._max_features_range,
            "n_threads": self._n_threads,
        }
        
        self._algorithm = algorithm.RandomForest(x, y,algorithm_params=parameters_dict)
                                                       
        parameters_dict = {
            "n_iterations": self._n_iterations,
            "test_size": self._test_size,
            "n_threads": self._n_threads,
            "splits_indices": self._splits_indices,
            "inner_cv": self._inner_cv,
        }
      
        self._validation = validation.RepeatedHoldOut(self._algorithm,validation_params=parameters_dict)
        
        classifier, best_params, results = self._validation.validate(y)

        
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, classifier_dir,self._output_dir)

        self._validation.save_results(self._output_dir)
        

def rf_classifications(model_name, columns, data_tsv_template, output_dir, months, indices_template=None,n_iterations=250,
                       test_size=0.2, n_threads=40, balanced=True, n_estimators_range=100, max_depth_range=5,
                       min_samples_split_range=2, max_features_range='auto', inner_cv=False):
   
    for i in months:
        if indices_template is None:splits_indices=None
        else:
            with open(indices_template, 'rb') as ind:splits_indices = pickle.load(ind,encoding='iso-8859-1')

        classification_dir = os.path.join(output_dir, '%s_months' % i, model_name)

        if not os.path.exists(classification_dir): os.makedirs(classification_dir)

        print("Running %s" % classification_dir)
        
        wf = TsvRFWf(data_tsv_template, columns, classification_dir, n_threads=n_threads, n_iterations=n_iterations,
                     test_size=test_size, balanced=balanced, n_estimators_range=n_estimators_range,
                     max_depth_range=max_depth_range, min_samples_split_range=min_samples_split_range,
                     max_features_range=max_features_range, splits_indices=splits_indices, inner_cv=inner_cv)
        
        wf.run()


def xgboost(file,columns,splits_indices,output):
    dataframe = pd.io.parsers.read_csv(file, sep='\t')
    x = dataframe[columns].to_numpy()
    unique = list(set(dataframe["diagnosis"]))
    y = np.array([unique.index(x) for x in dataframe["diagnosis"]])
    
    #apply random forest from algorithm on the parameters given by the user 
    parameters_dict = {
            "balanced": True,
            "grid_search_folds": 10,
            "max_depth_range": 10,
            "learning_rate_range": 0.01,
            "n_estimators_range": 150,
            "colsample_bytree_range": 0.5,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "n_threads": 20,
        }
    
    algorithm1 = algorithm.XGBoost(x,y,parameters_dict)
                                                    
    
    
    parameters_dict = {
        "n_iterations": 250,
        "test_size": 0.2,
        "n_threads": 20,
        "splits_indices": splits_indices,
        "inner_cv": False,
    }
    
    validation1 = validation.RepeatedHoldOut(algorithm1,validation_params=parameters_dict)
    classifier, best_params, results = validation1.validate(y)    
    
    classifier_dir = os.path.join(output, 'classifier')
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    algorithm1.save_classifier(classifier, classifier_dir)
    algorithm1.save_parameters(best_params, classifier_dir)
    weights = algorithm1.save_weights(classifier, classifier_dir,output)

    validation1.save_results(output)

def lreg(file,columns,splits_indices,output):

    dataframe = pd.io.parsers.read_csv(file, sep='\t')
    x = dataframe[columns].to_numpy()
    unique = list(set(dataframe["diagnosis"]))
    y = np.array([unique.index(x) for x in dataframe["diagnosis"]])
    
    #apply random forest from algorithm on the parameters given by the user 
    parameters_dict = {
            "penalty": "l2",
            "balanced": True,
            "grid_search_folds": 10,
            "c_range": np.logspace(-6, 2, 17),
            "n_threads": 20,
        }
    
    algorithm1 = algorithm.LogisticReg(x,y,parameters_dict)
                                                    
    
    
    parameters_dict = {
        "n_iterations": 250,
        "test_size": 0.2,
        "n_threads": 20,
        "splits_indices": splits_indices,
        "inner_cv": True,
    }
    
    validation1 = validation.RepeatedHoldOut(algorithm1,validation_params=parameters_dict)
    
    classifier, best_params, results = validation1.validate(y)

    
    
    classifier_dir = os.path.join(output, 'classifier')
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    algorithm1.save_classifier(classifier, classifier_dir)
    algorithm1.save_parameters(best_params, classifier_dir)
    weights = algorithm1.save_weights(classifier, classifier_dir,output)

    validation1.save_results(output)