from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and feature columns
model = joblib.load('models/driver_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# Global variable to store current data
current_data = None

# Function to calculate driver stats
def calculate_driver_stats(data):
    stats = {}
    for driver in data['Suggested_Driver'].unique():
        driver_data = data[data['Suggested_Driver'] == driver]
        stats[driver] = {
            'total_assignments': len(driver_data),
            'average_distance': driver_data['Distance'].mean(),  # Replace with actual distance calculations
            'average_duration': driver_data['Duration'].mean()  # Replace with actual duration calculations
        }
    return stats

# Route for the upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        data = pd.read_csv(file)
        
        # Preprocess the input data similar to training
        data['PICK_UP_HOUR'] = data['PICK_UP_TIME'].str[:-1].astype(int)
        data['PICK_UP_PERIOD'] = data['PICK_UP_TIME'].str[-1]
        data['CITY_PAIR'] = data['PICKUP_CITY'] + "_" + data['DROP_OFF_CITY']
        data_encoded = pd.get_dummies(data, columns=['PICK_UP_PERIOD', 'PICKUP_CITY', 'DROP_OFF_CITY', 'CITY_PAIR'], drop_first=True)
        
        # Ensure the test data has the same columns as the training data
        for col in feature_columns:
            if col not in data_encoded:
                data_encoded[col] = 0
        data_encoded = data_encoded[feature_columns]
        
        # Predict using the model
        data['Suggested_Driver'] = model.predict(data_encoded)
        drivers = data['Suggested_Driver'].unique().tolist()
        data['Distance'] = 10  # Placeholder distance, calculate actual distances
        data['Duration'] = 15  # Placeholder duration, calculate actual durations
        driver_stats = calculate_driver_stats(data)
        
        # Store current data for editing
        current_data = data
        
        return render_template('index.html', data=data.to_dict(orient='records'), drivers=drivers, driver_stats=driver_stats)

# Route for saving changes
@app.route('/save', methods=['POST'])
def save_changes():
    global current_data
    if current_data is None:
        return redirect(url_for('index'))
    
    # Capture form data and update driver assignments using hidden inputs for original index
    for i, row in current_data.iterrows():
        original_index = request.form.get(f'original_index_{row.name}')
        selected_driver = request.form.get(f'driver_{original_index}')
        
        # Validate that original_index and selected_driver are not None or empty
        if original_index is not None and original_index != '' and selected_driver:
            try:
                # Update the driver assignment in the DataFrame
                current_data.at[int(original_index), 'Suggested_Driver'] = selected_driver
            except ValueError:
                continue  # Skip this update if there's an issue with conversion

    # Save the updated data to a CSV file
    current_data.to_csv('data/modified_data.csv', index=False)
    
    # Return to the same page to allow further edits or download
    return redirect(url_for('index'))

# Route for downloading the modified file
@app.route('/download', methods=['GET'])
def download_file():
    path = 'data/modified_data.csv'
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

