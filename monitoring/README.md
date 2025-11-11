# 📊 Evidently AI Monitoring Dashboard

This folder contains comprehensive model monitoring tools using Evidently AI for the Road Accident Risk Prediction pipeline.

## 📁 Contents

### `evidently_dashboard.ipynb`
Interactive Jupyter Notebook for detailed model monitoring analysis with:
- **Data Drift Detection**: Track changes in feature distributions between training and production data
- **Model Performance Monitoring**: Regression metrics comparison (MAE, RMSE, R², MAPE)
- **Target Drift Analysis**: Monitor shifts in target variable distribution
- **Interactive Visualizations**: Explore detailed drift reports with plots and statistics
- **Statistical Comparisons**: Feature-wise analysis of mean and std changes

## 🚀 How to Use

### 1. **Open Jupyter Notebook**
```powershell
# Activate environment
.\mlenv\Scripts\activate

# Start Jupyter Lab/Notebook
jupyter lab monitoring/evidently_dashboard.ipynb
# OR
jupyter notebook monitoring/evidently_dashboard.ipynb
```

### 2. **Run All Cells**
The notebook will:
- Load training (reference) and test (current) data
- Generate predictions using trained model
- Create 3 interactive HTML reports:
  - `regression_performance_report.html` - Model performance metrics
  - `data_drift_report.html` - Feature distribution changes
  - `target_drift_report.html` - Target variable drift
- Extract and display drift metrics
- Provide actionable recommendations

### 3. **View Interactive Reports**
The generated HTML reports are saved in `artifacts/monitoring/` and can be opened directly in any browser:
```
artifacts/monitoring/
├── regression_performance_report.html  # Open in browser
├── data_drift_report.html              # Open in browser
├── target_drift_report.html            # Open in browser
└── evidently_report.txt                # Text summary
```

## 📊 Understanding the Reports

### **Regression Performance Report**
- **Metrics**: MAE, RMSE, R², MAPE comparisons
- **Error Distribution**: Visualize prediction errors
- **Residuals Analysis**: Check for patterns in errors
- **Top Errors**: Identify worst predictions

### **Data Drift Report**
- **Dataset Drift**: Overall drift status (YES/NO)
- **Feature-wise Drift**: Individual feature drift scores
- **Distribution Plots**: Compare reference vs current distributions
- **Statistical Tests**: P-values and test statistics
- **Drift Share**: Percentage of drifted features

### **Target Drift Report**
- **Target Distribution**: Compare target variable distributions
- **Statistical Tests**: Detect significant changes
- **Correlation Analysis**: Check feature-target relationships
- **Prediction Drift**: Monitor prediction distributions

## 🎯 Key Metrics to Monitor

1. **Dataset Drift**: If `True`, significant drift detected → Consider retraining
2. **Drift Share**: `> 30%` → High drift, investigate immediately
3. **R² Score**: If drops `> 5%` → Model performance degraded
4. **RMSE**: If increases `> 10%` → Prediction accuracy worsened

## 🔄 Integration with Pipeline

The monitoring component is automatically integrated into the pipeline:

```python
# In main.py - Stage 7
from heartpipeline.pipeline.stage_07_monitoring import ModelMonitoringPipeline

monitoring = ModelMonitoringPipeline()
report_paths = monitoring.main()
```

This generates all reports automatically after model training and evaluation.

## 📈 Evidently Cloud Integration (Optional)

For production monitoring with centralized dashboard:

1. **Sign up**: https://app.evidentlyai.com/
2. **Create Project**: Set up monitoring project
3. **Connect Pipeline**: Use Evidently Cloud API
4. **View Dashboard**: Track metrics over time with automated alerts

### Example Cloud Integration:
```python
from evidently.ui.workspace import Workspace

# Connect to Evidently Cloud
ws = Workspace.create("path/to/workspace")
project = ws.create_project("Road Accident Risk Monitoring")

# Add reports to project
project.add_report(regression_report)
project.add_report(data_drift_report)

# View dashboard
project.dashboard()
```

## 🛠️ Customization

### Add Custom Metrics
```python
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

custom_report = Report(metrics=[
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name='speed'),
    # Add more custom metrics
])
```

### Schedule Automated Monitoring
```python
# In production, run monitoring daily/weekly
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()
scheduler.add_job(
    func=generate_monitoring_reports,
    trigger="cron",
    day_of_week="mon",
    hour=9
)
scheduler.start()
```

## 📚 Resources

- **Evidently Documentation**: https://docs.evidentlyai.com/
- **Dashboard Guide**: https://docs.evidentlyai.com/docs/platform/dashboard_overview
- **GitHub Examples**: https://github.com/evidentlyai/evidently/tree/main/examples
- **Colab Tutorial**: https://colab.research.google.com/drive/1bdXdxffXo8Ag0Wbyu2d9agXaUCnZjFDm

## ⚠️ Troubleshooting

### Issue: Import Errors
```bash
# Reinstall evidently
pip install evidently --upgrade
```

### Issue: Reports Not Displaying
```bash
# Ensure Jupyter extensions are installed
jupyter nbextension enable --py widgetsnbextension
```

### Issue: Large HTML Files
```bash
# Use JSON format for large datasets
report.save_json('report.json')
```

## 🎉 Next Steps

1. **Explore Notebook**: Run `evidently_dashboard.ipynb` to see all visualizations
2. **View HTML Reports**: Open generated HTML files in browser
3. **Set Alerts**: Configure drift thresholds for automated alerts
4. **Integrate CI/CD**: Add monitoring checks to deployment pipeline
5. **Track Over Time**: Log reports to database for historical analysis

---

**Created for**: ML Pipeline with Evidently AI  
**Author**: Road Accident Risk Prediction Project  
**Last Updated**: 2025-11-11  
