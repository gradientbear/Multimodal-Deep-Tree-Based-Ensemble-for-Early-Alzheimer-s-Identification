from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from preprocess import get_data, data_models, all_features
from models import get_TabNet_model

# Load data
data = get_data()

# Train and evaluate TabNet model
for model_name, features in data_models.items():
    model = get_TabNet_model()
    print(f"Model: {model_name}, Features: {features}")
    X = data[features]
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}, MSE: {mse:.2f}")
    print("\n")
