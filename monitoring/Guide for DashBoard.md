# 🎯 How to View Evidently AI Monitoring Dashboard

## ✅ Quick Start - View Interactive HTML Reports

### Method 1: Open HTML Report Directly in Browser

1. **Navigate to the reports directory:**
   ```
   artifacts/monitoring/
   ```

2. **Open the HTML file:**
   - Double-click `data_drift_report.html`
   - OR right-click → "Open with" → Select your browser (Chrome, Firefox, Edge)

3. **Explore the Interactive Dashboard:**
   - View drift detection summary
   - Explore feature-by-feature drift analysis  
   - Interactive plots and statistical tests
   - Distribution comparisons

### Method 2: Use Python Script

Run the automated report generator:

```powershell
# Activate environment
.\mlenv\Scripts\activate

# Navigate to monitoring folder
cd monitoring

# Generate reports
python generate_reports.py
```

This will:
- Load training and test data
- Generate predictions
- Create interactive HTML drift report
- Open the report location for you

### Method 3: Use Jupyter Notebook (Full Analysis)

For comprehensive analysis with multiple reports:

```powershell
# Start Jupyter
jupyter lab monitoring/evidently_dashboard.ipynb
# OR
jupyter notebook monitoring/evidently_dashboard.ipynb
```

Run all cells to generate:
- Regression Performance Report
- Data Drift Report  
- Target Drift Report
- Statistical comparisons
- Feature drift analysis

## 📊 What You'll See in the Dashboard

### Data Drift Report Features:

1. **Summary Section:**
   - Dataset drift status (YES/NO)
   - Number of drifted features
   - Percentage of drifted columns

2. **Feature Analysis:**
   - Individual feature drift scores
   - P-values from statistical tests
   - Distribution plots (reference vs current)

3. **Interactive Visualizations:**
   - Histogram comparisons
   - Scatter plots
   - Drift heatmaps
   - Feature importance for drift

4. **Statistical Tests:**
   - Kolmogorov-Smirnov test for numerical features
   - Chi-squared test for categorical features
   - Jensen-Shannon divergence

## 🔍 Understanding the Metrics

### Drift Score
- **0.0 - 0.1**: No drift (🟢 Safe)
- **0.1 - 0.3**: Minor drift (🟡 Monitor)
- **0.3 - 0.5**: Moderate drift (🟠 Investigate)
- **> 0.5**: Significant drift (🔴 Action Required)

### Dataset Drift
- **NO**: Less than 30% of features drifted → Continue monitoring
- **YES**: More than 30% of features drifted → Consider retraining

## 🚀 Advanced: Evidently Cloud Dashboard

For production monitoring with centralized dashboard:

### 1. Sign Up for Evidently Cloud
```
https://app.evidentlyai.com/
```

### 2. Create a Project
- Log in to Evidently Cloud
- Click "New Project"
- Name: "Road Accident Risk Monitoring"

### 3. Connect Your Pipeline

Update `monitoring.py` or create a new script:

```python
from evidently.ui.workspace import Workspace
from evidently import Report
from evidently.presets import DataDriftPreset

# Connect to Evidently Cloud
ws = Workspace.create("workspace")
project = ws.create_project("Road Accident Risk")

# Generate and upload report
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(current_data=test_data, reference_data=train_data)

# Add to project
project.add_snapshot(snapshot)

# View dashboard
project.dashboard()
```

### 4. View Centralized Dashboard
- Track metrics over time
- Set up automated alerts
- Compare multiple model versions
- Historical trend analysis

## 📈 Production Monitoring Setup

### Schedule Automated Reports

Create `scheduled_monitoring.py`:

```python
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

def generate_reports():
    """Generate monitoring reports"""
    subprocess.run(["python", "monitoring/generate_reports.py"])

scheduler = BlockingScheduler()

# Run every day at 9 AM
scheduler.add_job(
    func=generate_reports,
    trigger="cron",
    day_of_week="mon-fri",
    hour=9,
    minute=0
)

print("🕐 Scheduler started. Reports will generate daily at 9 AM")
scheduler.start()
```

### Set Up Drift Alerts

Add to your monitoring component:

```python
def check_drift_threshold(drift_ratio, threshold=0.3):
    """Check if drift exceeds threshold and send alert"""
    if drift_ratio > threshold:
        # Send email/Slack notification
        send_alert(f"⚠️ Drift detected: {drift_ratio:.1%} of features drifted!")
        return True
    return False
```

## 📁 File Locations

```
ML Pipeline (Evidently)/
├── artifacts/
│   └── monitoring/
│       ├── data_drift_report.html         ← OPEN THIS IN BROWSER
│       └── evidently_report.txt            ← Text summary
├── monitoring/
│   ├── generate_reports.py                 ← Run this script
│   ├── evidently_dashboard.ipynb           ← Jupyter notebook
│   └── README.md                           ← This file
└── src/
    └── heartpipeline/
        └── components/
            └── monitoring.py               ← Pipeline integration
```

## 🎓 Resources

### Official Documentation
- **Evidently Docs**: https://docs.evidentlyai.com/
- **Dashboard Guide**: https://docs.evidentlyai.com/docs/platform/dashboard_overview
- **Metrics Reference**: https://docs.evidentlyai.com/reference/all-metrics

### Tutorials
- **Colab Example**: https://colab.research.google.com/drive/1bdXdxffXo8Ag0Wbyu2d9agXaUCnZjFDm
- **GitHub Examples**: https://github.com/evidentlyai/evidently/tree/main/examples
- **Video Tutorials**: https://www.youtube.com/@EvidentlyAI

### Community
- **GitHub**: https://github.com/evidentlyai/evidently
- **Discord**: https://discord.gg/xZjKRaNp8b
- **Blog**: https://www.evidentlyai.com/blog

## ❓ Troubleshooting

### Issue: HTML file won't open
**Solution**: Try different browser or check file permissions

### Issue: Plots not displaying
**Solution**: Ensure JavaScript is enabled in browser

### Issue: "No module named 'evidently'"
**Solution**: 
```powershell
.\mlenv\Scripts\activate
pip install evidently
```

### Issue: Unicode errors in text report
**Solution**: Open with UTF-8 encoding:
```python
with open('report.txt', 'r', encoding='utf-8') as f:
    content = f.read()
```

## 🎉 Next Steps

1. ✅ **View Current Report**: Open `artifacts/monitoring/data_drift_report.html`
2. 📊 **Run Jupyter Notebook**: Explore `monitoring/evidently_dashboard.ipynb`
3. 🔄 **Automate**: Set up scheduled report generation
4. ☁️ **Scale**: Integrate with Evidently Cloud for production
5. 📧 **Alert**: Configure drift threshold alerts

---

**Created**: 2025-11-11  
**Pipeline**: Road Accident Risk Prediction  
**Framework**: Evidently AI 0.7.15  
