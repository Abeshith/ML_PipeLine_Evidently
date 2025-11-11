from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

# Load model and preprocessing artifacts
MODEL_PATH = Path("artifacts/model_trainer/model.pkl")
SCALER_PATH = Path("artifacts/data_transformation/scaler.pkl")
LABEL_ENCODERS_PATH = Path("artifacts/data_transformation/label_encoders.pkl")

# Load artifacts
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(LABEL_ENCODERS_PATH, 'rb') as f:
    label_encoders = pickle.load(f)

# Feature configuration
CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']

@app.route('/')
def index():
    """Home page with animations"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Extract form data with validation
        data = {
            'road_type': request.form.get('road_type', 'highway'),
            'num_lanes': int(request.form.get('num_lanes', 2)),
            'curvature': float(request.form.get('curvature', 0.2)),
            'speed_limit': int(request.form.get('speed_limit', 60)),
            'lighting': request.form.get('lighting', 'daylight'),
            'weather': request.form.get('weather', 'clear'),
            'road_signs_present': 1 if request.form.get('road_signs_present', 'yes') == 'yes' else 0,
            'public_road': 1 if request.form.get('public_road', 'yes') == 'yes' else 0,
            'time_of_day': request.form.get('time_of_day', 'morning'),
            'holiday': 1 if request.form.get('holiday', 'no') == 'yes' else 0,
            'school_season': 1 if request.form.get('school_season', 'yes') == 'yes' else 0,
            'num_reported_accidents': int(request.form.get('num_reported_accidents', 0))
        }
        
        # Validate no None values in categorical fields
        for field in ['road_type', 'lighting', 'weather', 'time_of_day']:
            if data[field] is None or data[field] == '':
                raise ValueError(f"Field '{field}' is required")
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Feature engineering (same as training)
        df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
        df['curvature_speed'] = df['curvature'] * df['speed_limit']
        df['lanes_curvature'] = df['num_lanes'] * df['curvature']
        df['high_speed'] = (df['speed_limit'] > 60).astype(int)
        df['high_curvature'] = (df['curvature'] > 0.5).astype(int)
        df['few_lanes'] = (df['num_lanes'] <= 2).astype(int)
        df['no_signs'] = (df['road_signs_present'] == 0).astype(int)
        df['holiday_risk'] = (df['holiday'] == 1).astype(int)
        
        # Speed categories - Match ACTUAL encoded values (only 3 categories exist)
        # The training data didn't have speeds > 80, so only low/medium/high exist
        speed_bins = [0, 40, 60, 120]
        df['speed_category'] = pd.cut(df['speed_limit'], bins=speed_bins, labels=['low', 'medium', 'high'])
        
        # Curvature categories - Match ACTUAL encoded values
        curv_bins = [-0.1, 0.3, 0.6, 1.0]
        df['curvature_category'] = pd.cut(df['curvature'], bins=curv_bins, labels=['low', 'medium', 'high'])
        
        # Encode ALL categorical features (including speed_category and curvature_category)
        all_categorical = CATEGORICAL_FEATURES + ['speed_category', 'curvature_category']
        for col in all_categorical:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Risk level classification
        if prediction < 0.3:
            risk_level = "Low Risk"
            risk_color = "#4CAF50"
            risk_icon = "✅"
            risk_msg = "Road conditions are safe for travel"
        elif prediction < 0.6:
            risk_level = "Medium Risk"
            risk_color = "#FF9800"
            risk_icon = "⚠️"
            risk_msg = "Exercise caution while driving"
        else:
            risk_level = "High Risk"
            risk_color = "#F44336"
            risk_icon = "🚨"
            risk_msg = "Dangerous conditions - avoid if possible"
        
        result = {
            'prediction': round(prediction, 4),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_icon': risk_icon,
            'risk_message': risk_msg,
            'confidence': round((1 - abs(prediction - 0.5)) * 100, 2)
        }
        
        return render_template('predict.html', result=result, input_data=data)
    
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        return render_template('predict.html', error=error_msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Ensure required fields exist with defaults
        defaults = {
            'road_type': 'highway', 'lighting': 'daylight', 'weather': 'clear',
            'time_of_day': 'morning', 'num_lanes': 2, 'curvature': 0.2,
            'speed_limit': 60, 'road_signs_present': 1, 'public_road': 1,
            'holiday': 0, 'school_season': 1, 'num_reported_accidents': 0
        }
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
        
        df = pd.DataFrame([data])
        
        # Feature engineering
        df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
        df['curvature_speed'] = df['curvature'] * df['speed_limit']
        df['lanes_curvature'] = df['num_lanes'] * df['curvature']
        df['high_speed'] = (df['speed_limit'] > 60).astype(int)
        df['high_curvature'] = (df['curvature'] > 0.5).astype(int)
        df['few_lanes'] = (df['num_lanes'] <= 2).astype(int)
        df['no_signs'] = (df['road_signs_present'] == 0).astype(int)
        df['holiday_risk'] = (df['holiday'] == 1).astype(int)
        
        # Speed and curvature categories - Match ACTUAL encoded values
        df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 40, 60, 120], labels=['low', 'medium', 'high'])
        df['curvature_category'] = pd.cut(df['curvature'], bins=[-0.1, 0.3, 0.6, 1.0], labels=['low', 'medium', 'high'])
        
        # Encode ALL categorical (including speed_category and curvature_category)
        all_categorical = CATEGORICAL_FEATURES + ['speed_category', 'curvature_category']
        for col in all_categorical:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        scaled_features = scaler.transform(df)
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'risk_level': 'Low' if prediction < 0.3 else 'Medium' if prediction < 0.6 else 'High'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    """Dashboard with monitoring reports"""
    return render_template('dashboard.html')

@app.route('/reports/drift')
def drift_report():
    """Serve the data drift HTML report"""
    try:
        with open('artifacts/monitoring/data_drift_report.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Drift report not found. Please run the monitoring pipeline first.", 404

@app.route('/reports/performance')
def performance_report():
    """Serve the performance report as text"""
    try:
        with open('artifacts/monitoring/evidently_report.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        return f"<pre style='padding: 20px; background: #f5f5f5; font-family: monospace;'>{content}</pre>"
    except FileNotFoundError:
        return "Performance report not found. Please run the monitoring pipeline first.", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
