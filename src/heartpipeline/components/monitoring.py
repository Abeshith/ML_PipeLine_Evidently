import os
import sys
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.metrics.regression import RegressionQualityMetric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import MonitoringConfig


class ModelMonitoring:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        os.makedirs(self.config.root_dir, exist_ok=True)

    def load_model(self):
        try:
            with open(self.config.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {self.config.model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            reference_data = pd.read_csv(self.config.reference_data_path)
            current_data = pd.read_csv(self.config.current_data_path)
            
            logger.info(f"Reference data shape: {reference_data.shape}")
            logger.info(f"Current data shape: {current_data.shape}")
            
            return reference_data, current_data
        except Exception as e:
            raise CustomException(e, sys)

    def generate_report(self):
        try:
            logger.info("Generating Evidently monitoring reports...")
            
            model = self.load_model()
            reference_data, current_data = self.load_data()
            if self.config.target_column in reference_data.columns:
                X_ref = reference_data.drop(self.config.target_column, axis=1)
                X_cur = current_data.drop(self.config.target_column, axis=1)
            else:
                X_ref = reference_data
                X_cur = current_data
            
            reference_data['prediction'] = model.predict(X_ref)
            current_data['prediction'] = model.predict(X_cur)
            
            logger.info(f"Reference data shape: {reference_data.shape}")
            logger.info(f"Current data shape: {current_data.shape}")
            
            numerical_features = X_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_ref.select_dtypes(include=['object', 'category']).columns.tolist()
            logger.info("Generating Data Drift Report...")
            data_drift_report = Report(metrics=[DataDriftPreset()])
            drift_snapshot = data_drift_report.run(
                current_data=current_data,
                reference_data=reference_data
            )
            drift_html = os.path.join(self.config.root_dir, 'data_drift_report.html')
            drift_snapshot.save_html(drift_html)
            logger.info(f"Data drift report saved: {drift_html}")
            ref_mean = reference_data.select_dtypes(include=['number']).mean()
            cur_mean = current_data.select_dtypes(include=['number']).mean()
            drift_pct = ((cur_mean - ref_mean) / ref_mean * 100).abs()
            
            ref_mae = mean_absolute_error(reference_data[self.config.target_column], reference_data['prediction'])
            ref_rmse = np.sqrt(mean_squared_error(reference_data[self.config.target_column], reference_data['prediction']))
            ref_r2 = r2_score(reference_data[self.config.target_column], reference_data['prediction'])
            
            cur_mae = mean_absolute_error(current_data[self.config.target_column], current_data['prediction'])
            cur_rmse = np.sqrt(mean_squared_error(current_data[self.config.target_column], current_data['prediction']))
            cur_r2 = r2_score(current_data[self.config.target_column], current_data['prediction'])
            
            significant_drift_features = (drift_pct > 10).sum()
            total_features = len(drift_pct)
            drift_ratio = significant_drift_features / total_features if total_features > 0 else 0
            dataset_drift = drift_ratio > 0.3
            
            with open(self.config.evidently_report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EVIDENTLY AI - MODEL MONITORING SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("DATASET INFORMATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Reference Data Size: {len(reference_data)} rows × {len(reference_data.columns)} columns\n")
                f.write(f"Current Data Size: {len(current_data)} rows × {len(current_data.columns)} columns\n")
                f.write(f"Target Column: {self.config.target_column}\n")
                f.write(f"Numerical Features: {len(numerical_features)}\n")
                f.write(f"Categorical Features: {len(categorical_features)}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("MODEL PERFORMANCE COMPARISON\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'Metric':<20} {'Reference':<15} {'Current':<15} {'Change %':<15}\n")
                f.write("-" * 80 + "\n")
                mae_change = ((cur_mae - ref_mae) / ref_mae * 100) if ref_mae != 0 else 0
                rmse_change = ((cur_rmse - ref_rmse) / ref_rmse * 100) if ref_rmse != 0 else 0
                r2_change = ((cur_r2 - ref_r2) / ref_r2 * 100) if ref_r2 != 0 else 0
                
                f.write(f"{'MAE':<20} {ref_mae:<15.4f} {cur_mae:<15.4f} {mae_change:>+14.2f}%\n")
                f.write(f"{'RMSE':<20} {ref_rmse:<15.4f} {cur_rmse:<15.4f} {rmse_change:>+14.2f}%\n")
                f.write(f"{'R² Score':<20} {ref_r2:<15.4f} {cur_r2:<15.4f} {r2_change:>+14.2f}%\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("DRIFT DETECTION SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Dataset Drift Detected: {'YES' if dataset_drift else 'NO'}\n")
                f.write(f"Features with >10% change: {significant_drift_features}/{total_features} ({drift_ratio:.1%})\n")
                f.write(f"Drift Threshold: 30% of features\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("FEATURE DRIFT ANALYSIS (Mean % Change)\n")
                f.write("=" * 80 + "\n")
                drift_sorted = drift_pct.sort_values(ascending=False)
                for col in drift_sorted.index[:15]:
                    status = "" if drift_sorted[col] > 10 else "" if drift_sorted[col] > 5 else ""
                    f.write(f"{col:.<50} {drift_sorted[col]:>6.2f}%\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("GENERATED REPORTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"1. Data Drift Report: {drift_html}\n")
                f.write(f"2. Text Summary: {self.config.evidently_report_path}\n\n")
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n")
                if dataset_drift:
                    f.write("Data drift detected! Consider:\n")
                    f.write("   - Retraining the model with recent data\n")
                    f.write("   - Investigating root causes of drift\n")
                    f.write("   - Adjusting feature engineering pipeline\n")
                    f.write("   - Setting up automated retraining triggers\n")
                else:
                    f.write("No significant drift detected\n")
                    f.write("   - Continue monitoring periodically\n")
                    f.write("   - Maintain current model in production\n")
                    f.write("   - Keep tracking performance metrics\n")
                
                if abs(r2_change) > 5:
                    f.write(f"\nR² score changed by {r2_change:+.1f}% - Monitor model performance closely!\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("HOW TO VIEW REPORTS\n")
                f.write("=" * 80 + "\n")
                f.write("1. Open HTML file in your browser for interactive visualizations:\n")
                f.write(f"   - Data Drift: {drift_html}\n")
                f.write("2. Use monitoring/evidently_dashboard.ipynb for detailed analysis\n")
                f.write("3. Use monitoring/generate_reports.py to regenerate reports\n")
                f.write("4. Set up automated report generation in production\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"All reports generated successfully!\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Summary report saved to {self.config.evidently_report_path}")
            logger.info(f"Monitoring report generated successfully!")
            
            return {
                'summary': str(self.config.evidently_report_path),
                'drift_html': drift_html,
                'dataset_drift': dataset_drift,
                'drift_ratio': drift_ratio,
                'performance': {
                    'reference': {'MAE': ref_mae, 'RMSE': ref_rmse, 'R2': ref_r2},
                    'current': {'MAE': cur_mae, 'RMSE': cur_rmse, 'R2': cur_r2}
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            logger.info("Falling back to basic statistics report...")
            
            try:
                model = self.load_model()
                reference_data, current_data = self.load_data()
                
                X_ref = reference_data.drop(self.config.target_column, axis=1) if self.config.target_column in reference_data.columns else reference_data
                X_cur = current_data.drop(self.config.target_column, axis=1) if self.config.target_column in current_data.columns else current_data
                
                reference_data['prediction'] = model.predict(X_ref)
                current_data['prediction'] = model.predict(X_cur)
                
                ref_mean = reference_data.select_dtypes(include=['number']).mean()
                cur_mean = current_data.select_dtypes(include=['number']).mean()
                drift_pct = ((cur_mean - ref_mean) / ref_mean * 100).abs()
                
                with open(self.config.evidently_report_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("BASIC MONITORING SUMMARY (Fallback Mode)\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Reference Data: {len(reference_data)} rows\n")
                    f.write(f"Current Data: {len(current_data)} rows\n\n")
                    f.write("Feature Drift (Mean % Change):\n")
                    f.write("-" * 80 + "\n")
                    for col in drift_pct.sort_values(ascending=False).index[:10]:
                        f.write(f"{col}: {drift_pct[col]:.2f}%\n")
                    f.write("\n" + "=" * 80 + "\n")
                
                logger.info(f"Basic report saved to {self.config.evidently_report_path}")
                return str(self.config.evidently_report_path)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                raise CustomException(e, sys)
