import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and preprocess the data
data = pd.read_csv('data/sample_data.csv')

# Example preprocessing (adjust based on actual data structure)
# Create features and target variable
data['PICK_UP_HOUR'] = data['PICK_UP_TIME'].str[:-1].astype(int)
data['PICK_UP_PERIOD'] = data['PICK_UP_TIME'].str[-1]
data['CITY_PAIR'] = data['PICKUP_CITY'] + "_" + data['DROP_OFF_CITY']

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['PICK_UP_PERIOD', 'PICKUP_CITY', 'DROP_OFF_CITY', 'CITY_PAIR'], drop_first=True)

# Prepare features and target
X = data_encoded.drop(columns=['DRIVER', 'CLIENTS_NAME', 'PICK_UP_TIME', 'APPT_TIME', 'PICK_UP_ADDRESS', 'DROP_OFF_ADDRESS'])
y = data_encoded['DRIVER']

# Save the feature columns to a file
joblib.dump(X.columns, 'models/feature_columns.pkl')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
#print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/driver_model.pkl')
