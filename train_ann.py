from models import get_ANN_model, TsvRFWf
from preprocess import data_models, all_features, get_data
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from clinica.pipelines.machine_learning import algorithm, validation



# Random Forest-based classification function
def rf_classifications(
    model_name, columns, data_tsv_template, output_dir, months, indices_template=None,
    n_iterations=250, test_size=0.2, n_threads=40, balanced=True,
    n_estimators_range=100, max_depth_range=5, min_samples_split_range=2,
    max_features_range='auto', inner_cv=False
):
    """
    Perform Random Forest-based classification with specific parameters.

    Args:
        model_name (str): Name of the model.
        columns (list): List of feature columns.
        data_tsv_template (pd.DataFrame): Data to be used for training.
        output_dir (str): Directory to save the results.
        months (list): List of month intervals to process.
        indices_template (str): Path to precomputed split indices.
        Other args: Various hyperparameters for RF training.
    """
    for i in months:
        # Load split indices if provided
        splits_indices = None
        if indices_template:
            with open(indices_template, 'rb') as ind:
                splits_indices = pickle.load(ind, encoding='iso-8859-1')

        # Create directory for classification results
        classification_dir = os.path.join(output_dir, f'{i}_months', model_name)
        os.makedirs(classification_dir, exist_ok=True)
        print(f"Running {classification_dir}")

        # Instantiate and run the classification workflow
        wf = TsvRFWf(
            data_tsv_template, columns, classification_dir,
            n_threads=n_threads, n_iterations=n_iterations, test_size=test_size,
            balanced=balanced, n_estimators_range=n_estimators_range,
            max_depth_range=max_depth_range, min_samples_split_range=min_samples_split_range,
            max_features_range=max_features_range, splits_indices=splits_indices, inner_cv=inner_cv
        )
        wf.run()


# XGBoost-based classification function
def xgboost(file, columns, splits_indices, output):
    """
    Train and validate an XGBoost model using the provided data.

    Args:
        file (str): Path to the input file.
        columns (list): List of feature columns.
        splits_indices (list): Precomputed split indices.
        output (str): Directory to save results.
    """
    dataframe = pd.read_csv(file, sep='\t')
    x = dataframe[columns].to_numpy()
    unique = list(set(dataframe["diagnosis"]))
    y = np.array([unique.index(x) for x in dataframe["diagnosis"]])

    # Define XGBoost hyperparameters
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

    # Train the XGBoost model
    algorithm1 = algorithm.XGBoost(x, y, parameters_dict)

    # Validation parameters
    validation_params = {
        "n_iterations": 250,
        "test_size": 0.2,
        "n_threads": 20,
        "splits_indices": splits_indices,
        "inner_cv": False,
    }

    validation1 = validation.RepeatedHoldOut(algorithm1, validation_params)
    classifier, best_params, results = validation1.validate(y)

    # Save results
    classifier_dir = os.path.join(output, 'classifier')
    os.makedirs(classifier_dir, exist_ok=True)
    algorithm1.save_classifier(classifier, classifier_dir)
    algorithm1.save_parameters(best_params, classifier_dir)
    algorithm1.save_weights(classifier, classifier_dir, output)
    validation1.save_results(output)


# Logistic Regression-based classification function
def lreg(file, columns, splits_indices, output):
    """
    Train and validate a Logistic Regression model.

    Args:
        file (str): Path to the input file.
        columns (list): List of feature columns.
        splits_indices (list): Precomputed split indices.
        output (str): Directory to save results.
    """
    dataframe = pd.read_csv(file, sep='\t')
    x = dataframe[columns].to_numpy()
    unique = list(set(dataframe["diagnosis"]))
    y = np.array([unique.index(x) for x in dataframe["diagnosis"]])

    # Define Logistic Regression hyperparameters
    parameters_dict = {
        "penalty": "l2",
        "balanced": True,
        "grid_search_folds": 10,
        "c_range": np.logspace(-6, 2, 17),
        "n_threads": 20,
    }

    # Train the Logistic Regression model
    algorithm1 = algorithm.LogisticReg(x, y, parameters_dict)

    # Validation parameters
    validation_params = {
        "n_iterations": 250,
        "test_size": 0.2,
        "n_threads": 20,
        "splits_indices": splits_indices,
        "inner_cv": True,
    }

    validation1 = validation.RepeatedHoldOut(algorithm1, validation_params)
    classifier, best_params, results = validation1.validate(y)

    # Save results
    classifier_dir = os.path.join(output, 'classifier')
    os.makedirs(classifier_dir, exist_ok=True)
    algorithm1.save_classifier(classifier, classifier_dir)
    algorithm1.save_parameters(best_params, classifier_dir)
    algorithm1.save_weights(classifier, classifier_dir, output)
    validation1.save_results(output)


# Artificial Neural Network (ANN) model training
classification_dir = os.path.join('output')
os.makedirs(classification_dir, exist_ok=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
])

# Compile the ANN model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Custom callback to save the best model
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SaveBestModel, self).__init__()
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}")
        if logs['val_accuracy'] > self.best:
            self.best = logs['val_accuracy']
            model.save_weights(os.path.join(classification_dir, 'best_model.h5'))


# Main loop for training models with ANN and RF classifications
output_dir = 'outputdir'
n_threads = 8
months = [36]
callbacks = [SaveBestModel()]

# Load data
data = get_data()

for i in data_models:
    print(f'Model: {data_models[i]}')
    x = data[data_models[i]].to_numpy()
    y = data['diagnosis'].to_numpy()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the ANN model
    model.fit(X_train, y_train, epochs=200, callbacks=callbacks, validation_split=0.2)

    # Load the best ANN model and generate features
    best_model = tf.keras.models.load_model('best_model.h5')
    features = best_model.predict(X_train)

    # Save the features for RF classification
    feature_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    new_df = pd.DataFrame(features, columns=feature_columns)
    new_df['diagnosis'] = y_train

    # Perform RF classification on the ANN-extracted features
    rf_classifications(i, feature_columns, new_df, output_dir, months, n_threads=n_threads)
