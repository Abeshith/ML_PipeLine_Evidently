import pandas as pd
import pickle
import os
from evidently import Report
from evidently.presets import DataDriftPreset

print("="*80)
print("EVIDENTLY AI - MONITORING REPORT GENERATOR")
print("="*80)

print("\nLoading data...")
train_data = pd.read_csv('../artifacts/data_transformation/train.csv')
test_data = pd.read_csv('../artifacts/data_transformation/test.csv')

print("Loading trained model...")
with open('../artifacts/model_trainer/model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Generating predictions...")
X_train = train_data.drop('accident_risk', axis=1)
X_test = test_data.drop('accident_risk', axis=1)

train_data['prediction'] = model.predict(X_train)
test_data['prediction'] = model.predict(X_test)

print(f"Training data: {train_data.shape}")
print(f"Test data: {test_data.shape}")

os.makedirs('../artifacts/monitoring', exist_ok=True)

print("\nGenerating Data Drift Report...")
data_drift_report = Report(metrics=[DataDriftPreset()])
snapshot = data_drift_report.run(
    current_data=test_data,
    reference_data=train_data
)
drift_html = '../artifacts/monitoring/data_drift_report.html'
snapshot.save_html(drift_html)
print(f"Saved: {drift_html}")

print("\n"+"="*80)
print("SUCCESS! Open the HTML file in your browser:")
print(f"   {os.path.abspath(drift_html)}")
print("="*80)
