from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from preprocess import get_data, data_models, all_features
from models import get_ML_classifiers

# Load data
data = get_data()

# Train and evaluate classifiers
for model_name, features in data_models.items():
    classifiers = get_ML_classifiers()
    print(f"Model: {model_name}, Features: {features}")
    X = data[features]
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    for classifier_name, classifier in classifiers.items():
        #print(f"Training {classifier_name}")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results[classifier_name] = [accuracy_score(y_test, y_pred), mean_squared_error(y_test, y_pred)]

    print(f"{'Classifier':<20}{'Accuracy':<10}{'MSE':<10}")
    print("-" * 40)
    for classifier_name, result in results.items():
        print(f"{classifier_name:<20}{result[0]:<10.2f}{result[1]:<10.2f}")
    print("-" * 40)
    print("\n")
