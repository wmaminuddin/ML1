# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import logging
from functools import lru_cache
from flask_cors import CORS
import os
from datetime import datetime

# Set up logging
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"{log_directory}/model_predictions_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Cache the model loading to improve performance
@lru_cache(maxsize=1)
def get_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Get the model and extract car makers
try:
    model = get_model()
    # Try to get classes from model if available
    try:
        car_makers = model.named_steps['classifier'].classes_.tolist()
        logging.info(f"Loaded {len(car_makers)} car classes from model")
    except (AttributeError, KeyError):
        # Fallback to hardcoded list if not available in model
        car_makers = [
            'PERODUA AXIA (AXIA - 1000 (AUTO))',
            'PERODUA ATIVA (ATIVA - 1000 TURBO AV (CVT))',
            'PERODUA MYVI (MYVI 1500 H (AUTO))',
            'PERODUA AXIA (AXIA - 1000 AV (AUTO))',
            'PERODUA BEZZA (BEZZA - 1300 AV (AUTO))',
            'PERODUA MYVI (MYVI 1500 AV (AUTO))',
            'PERODUA ATIVA (ATIVA - 1000 TURBO X (CVT))',
            'PERODUA BEZZA (BEZZA - 1300 X (AUTO))',
            'PERODUA MYVI (MYVI 1500 H (CVT))',
            'PERODUA MYVI (MYVI 1500 AV (CVT))',
            'PERODUA ATIVA (ATIVA - 1000 TURBO H (CVT))',
            'PERODUA AXIA (AXIA - 1000 SE (AUTO))',
            'PERODUA MYVI (MYVI 1300 X (AUTO))',
            'PERODUA BEZZA (BEZZA - 1000 G (AUTO))',
            'PERODUA MYVI (MYVI 1300 SE (AUTO))',
            'PERODUA ALZA (ALZA - 1500 AV (CVT))',
            'PERODUA MYVI (MYVI 1500 SE (AUTO))',
            'PERODUA ARUZ (ARUZ 1500 X (AUTO))',
            'PERODUA AXIA (AXIA - 1000 (MANUAL))',
            'PERODUA MYVI (MYVI 1300 G (CVT))',
            'PERODUA ALZA (ALZA 1500 (AUTO))',
            'PERODUA AXIA (AXIA - 1000 G (CVT))',
            'PERODUA ARUZ (ARUZ 1500 AV (AUTO))',
            'PERODUA AXIA (AXIA - 1000 STYLE (AUTO))',
            'PERODUA ALZA (ALZA 1500 AV (AUTO))',
            'PERODUA VIVA (VIVA  850 (MANUAL))',
            'PERODUA ALZA (ALZA 1500 SE (AUTO))',
            'PERODUA AXIA (AXIA - 1000 SE (CVT))',
            'PERODUA MYVI (MYVI 1500 X (CVT))',
            'PERODUA BEZZA (BEZZA - 1000 GXTRA (AUTO))',
            'PERODUA MYVI (MYVI 1300 (AUTO))',
            'PERODUA ALZA (ALZA - 1500 H (CVT))',
            'PERODUA ALZA (ALZA - 1500 X (CVT))',
            'PERODUA AXIA (AXIA - 1000 X (CVT))',
            'PERODUA AXIA (AXIA - 1000 AV (CVT))',
            'PERODUA VIVA (VIVA  1000 (MANUAL))',
            'PERODUA VIVA (VIVA 1000 (AUTO))',
            'PERODUA MYVI (MYVI 1.5L SE (AT))',
            'PERODUA ',
            'PERODUA AXIA (AXIA - 1000 E (MANUAL))',
            'PERODUA MYVI (MYVI 1.3 (MANUAL))',
            'PERODUA ALZA (ALZA 1500 (MANUAL ))',
            'PERODUA MYVI (MYVI 1500 Extreme (AUTO))',
            'PERODUA VIVA (VIVA  660 (MANUAL))',
            'PERODUA MYVI (MYVI 1.5L AV (AT))',
            'PERODUA BEZZA (BEZZA - 1000 G (MANUAL))',
            'PERODUA MYVI (MYVI 1.3 SE (MANUAL))',
            'PERODUA MYVI (MYVI 1500 Extreme (MANUAL))',
            'PERODUA MYVI ',
            'PERODUA BEZZA (BEZZA - 1300 X (MANUAL))',
            'PERODUA AXIA (AXIA - 1000 SE (MANUAL))',
            'PERODUA KELISA (KELISA 1.0 (AUTO))',
            'PERODUA MYVI (MYVI 1500 SE (MANUAL))',
            'PERODUA ALZA (ALZA 1500 SE (MANUAL))',
            'PERODUA BEZZA (BEZZA - 1000 GXTRA (MANUAL))',
            'PERODUA KELISA (KELISA 1.0 (MANUAL))',
            'PERODUA MYVI (MYVI 1.0 (MANUAL))',
            'PERODUA KANCIL (KANCIL 850 (MANUAL))',
            'PERODUA KANCIL (KANCIL 660 (MANUAL))',
            'PERODUA KENARI (KENARI (AUTO))'
        ]
        logging.info("Using hardcoded car classes list")
except Exception as e:
    logging.critical(f"Failed to initialize model: {e}")
    raise

# Create simplified car model names for display
simplified_car_makers = {}
for car in car_makers:
    parts = car.split('(')
    if len(parts) > 1:
        # Extract main model name and first specification
        main_model = parts[0].strip()
        spec = parts[1].split(')')[0].strip()
        simplified_car_makers[car] = f"{main_model} - {spec}"
    else:
        simplified_car_makers[car] = car.strip()

# Define valid ranges for input validation
VALID_RANGES = {
    'age': (18, 100),
    'gender': (0, 1),  # Assuming 0=Male, 1=Female
    'race': (0, 3),    # Assuming 0=Malay, 1=Chinese, 2=Indian, 3=Others
    'maritalStatus': (0, 4)  # Assuming 0=Single, 1=Married, 2=Divorced, 3=Widowed, 4=Others
}

# Get feature importance if available
try:
    feature_names = ['AGE'] + [f for f in model.named_steps['preprocessor'].get_feature_names_out() 
                               if f.startswith(('GENDER', 'RACE', 'MARITAL_STATUS'))]
    importances = model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    logging.info("Feature importance loaded successfully")
except Exception as e:
    importance_df = None
    logging.warning(f"Could not extract feature importance: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        
        # Extract and validate input values
        try:
            # Age should be float as in training
            age = float(data['age'])
            
            # Convert numeric codes to categorical strings
            gender_map = {0: 'Male', 1: 'Female'}
            race_map = {0: 'Malay', 1: 'Chinese', 2: 'Indian', 3: 'Others'}
            marital_map = {0: 'SINGLE', 1: 'MARRIED', 2: 'DIVORCE', 3: 'WIDOW', 4: 'OTHERS'}
            
            gender = gender_map[int(data['gender'])]
            race = race_map[int(data['race'])]
            marital_status = marital_map[int(data['maritalStatus'])]
            
        except (ValueError, KeyError) as e:
            error_msg = f'Invalid input data: {str(e)}'
            return jsonify({'error': error_msg}), 400

        # Create a pandas DataFrame with the correct column names and types
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [gender],
            'RACE': [race],
            'MARITAL_STATUS': [marital_status]
        })
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Set minimum confidence threshold (0%)
        min_confidence = 0
        
        # Sort predictions by probability
        top_indices = np.argsort(probabilities)[::-1]
        
        # Filter by confidence threshold and take top 5
        top_predictions = []
        for i in top_indices:
            if probabilities[i] >= min_confidence and len(top_predictions) < 5:
                top_predictions.append({
                    "carMaker": car_makers[i],
                    "displayName": simplified_car_makers[car_makers[i]],
                    "probability": round(probabilities[i] * 100, 2)
                })
        
        # Check if top_predictions is empty before accessing it
        if top_predictions:
            logging.info(f"Successful prediction for input: {input_data.to_dict()}, Top prediction: {top_predictions[0]['displayName']} ({top_predictions[0]['probability']}%)")
        else:
            logging.info(f"No predictions met threshold for input: {input_data.to_dict()}")
        
        return jsonify({
            'topPredictions': top_predictions,
            'userProfile': {
                'age': age,
                'gender': gender,
                'race': race,
                'maritalStatus': marital_status
            }
        })

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logging.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Return the feature importance data as JSON"""
    if importance_df is not None:
        return jsonify({
            'features': importance_df.to_dict(orient='records'),
            'description': 'Higher values indicate more important features for prediction'
        })
    else:
        return jsonify({
            'error': 'Feature importance data not available'
        }), 404

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Return information about the model"""
    try:
        # Get model metadata if available
        model_type = type(model.named_steps['classifier']).__name__
        n_estimators = getattr(model.named_steps['classifier'], 'n_estimators', 'Unknown')
        n_classes = len(car_makers)
        
        return jsonify({
            'model_type': model_type,
            'estimators': n_estimators,
            'classes': n_classes,
            'features': ['Age', 'Gender', 'Race', 'Marital Status'],
            'target': 'Car Model Preference'
        })
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        return jsonify({
            'error': 'Could not retrieve model information'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logging.info("Starting Car Purchase Predictor API")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
