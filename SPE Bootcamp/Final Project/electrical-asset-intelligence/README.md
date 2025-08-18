# README.md

## AI-Powered Prescriptive Electrical Asset Intelligence Platform



**A prescriptive maintenance system for electrical assets in midstream gas operations**

This project demonstrates advanced AI/ML techniques applied to transform electrical maintenance practices, developed for the SPE AI Machine Learning for Energy Professionals Boot Camp.

## 🎯 Project Overview

The system goes beyond prediction to provide **actionable maintenance recommendations** with quantified business impact. Built on real-world experience from electrical engineering internship at Bapco Gas, it transitions maintenance practices from reactive to prescriptive.

### **Key Features**
- **Multi-asset coverage**: Transformers, Motors, Capacitor Banks, UPS systems
- **Advanced AI stack**: LSTM autoencoders, Random Forest with SMOTE, Bayesian optimization
- **Prescriptive engine**: Prioritized recommendations with cost-benefit analysis
- **Business impact quantification**: ROI calculations for maintenance decisions

## 📦 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/electrical-asset-intelligence.git
cd electrical-asset-intelligence
```

### 2. Setup Environment
```bash
python -m venv env
source env/bin/activate  # Linux/MacOS
# OR
env\Scripts\activate     # Windows

pip install -r requirements.txt
pip install imbalanced-learn  # For SMOTE functionality
```

### 3. Generate Synthetic Data
```bash
python data/synthetic_data_generator.py
```
Creates realistic datasets for all asset types based on real maintenance experience.

### 4. Run Analysis Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🏗️ Project Structure

```
electrical-asset-intelligence/
├── README.md
├── Final_Project_Report.md
├── requirements.txt
├── __pycache__
├── data/
│ ├── synthetic_data_generator.py
│ ├── transformer_data.csv
│ ├── motor_data.csv
│ ├── capacitor_bank_data.csv
│ └── ups_data.csv
├── notebooks/
│ ├── data
│ ├── 01_data_exploration.ipynb
│ ├── 01_data_exploration.ipynb
│ ├── 02_anomaly_detection.ipynb
│ ├── 03_hyperparameter_tuning.ipynb
│ ├── 04_reinforcement_learning_scheduling.ipynb
│ └── 05_prescriptive_engine_demo.ipynb
├── src/
│ ├── data_preprocessing.py
│ ├── anomaly_detection.py
│ ├── health_classifier.py
│ └── prescriptive_engine.py
├── models/
│ ├── lstm_autoencoder_transformer.h5
│ ├── best_random_forest_transformer.pkl
│ └── [other models]
├── results/
│ ├── 1.%20Model%20Evaluation
│ ├── 2.%20Predictions
│ ├── 3.%20Prescriptive%20Actions
│ ├── 4.%20Business%20Impact
│ ├── 5.%20Asset%20Health
│ ├── 6.%20Data%20Exploration
│ └── .%20Reports%20&%20Analyses
└── ── ── ── ── ── ── ── ── ── ── ── ── ── 
```

## 🚀 Usage

### Data Preprocessing
```bash
python src/data_preprocessing.py
```

### Anomaly Detection
```bash
python src/anomaly_detection.py
```

### Health Classification
```bash
python src/health_classifier.py
```

### Prescriptive Engine
```bash
python src/prescriptive_engine.py
```

## 📊 Key Results

### **Comprehensive Results Overview**
The project generated extensive analytical outputs across seven key categories, providing deep insights into electrical asset health and maintenance optimization.

### **Performance Achievements**
- **95% accuracy** in predicting transformer health states
- **12.7% improvement** in Critical class recall with SMOTE
- **8.2% F1 score improvement** through hyperparameter tuning
- **5.1% anomaly detection rate** using LSTM autoencoders
- **Actionable recommendations** with quantified cost-benefit analysis

### **Sample Prescriptive Output**
```
Asset #42 - Transformer (Critical)
Recommendation: Replace phase B insulation
Urgency: Immediate | Cost: $3,500 | Confidence: 85%
Expected Value: $12,000 production loss prevented
Time Frame: within 24 hours | Downtime: 45-60 minutes
```

### **Business Impact Highlights**
- **Production Continuity**: Prevent unexpected shutdowns through proactive maintenance
- **Resource Optimization**: 30-40% improvement in maintenance team utilization
- **Cost Savings**: ROI of 3-5x through prescriptive recommendations
- **Safety Enhancement**: Early identification of critical failure modes

## 🔬 Technical Highlights

### **Advanced AI/ML Implementation**
1. **LSTM Autoencoders** for time-series anomaly detection in sensor data
2. **SMOTE oversampling** to handle rare failure events (12.7% recall improvement)
3. **Bayesian hyperparameter tuning** for optimal model performance
4. **Q-learning scheduling** for maintenance resource optimization
5. **Multi-asset prescriptive analytics** with business impact quantification

### **Real-World Foundation**
Built on authentic electrical maintenance experience at Bapco Gas, addressing:
- Transformer insulation degradation patterns
- Motor bearing failure signatures
- Capacitor bank harmonic distortion issues
- UPS battery degradation monitoring

### **Robust Results Architecture**
Comprehensive output organization covering:
- Model evaluation and performance benchmarking
- Failure probability predictions and risk distributions
- Prescriptive maintenance recommendations
- Business impact analysis and resource optimization
- Asset health monitoring and time series analysis
- Data exploration and statistical insights

## 📈 Performance Metrics

| Asset Type | Model | Accuracy | F1-Score | AUC | Key Features |
|------------|-------|----------|----------|-----|--------------|
| **Transformer** | Tuned RF + SMOTE | 0.95 | 0.93 | 0.99 | Insulation resistance, Partial discharge |
| **Motor** | Tuned Random Forest | 0.92 | 0.90 | 0.96 | Vibration patterns, Temperature differential |
| **Capacitor Bank** | Tuned Random Forest | 0.95 | 0.93 | 0.99 | Harmonic distortion, Thermal indicators |
| **UPS** | Tuned Random Forest | 0.93 | 0.91 | 0.97 | Voltage per cell, Temperature trends |

## 🎯 Why This Project Stands Out

### **1. Authentic Industry Experience**
- Built on **real maintenance challenges** at Bapco Gas midstream facility
- Addresses **specific failure modes** observed during electrical engineering internship
- Uses **actual parameters and procedures** from hands-on maintenance work

### **2. Comprehensive Asset Coverage**
- **Power Generation**: Transformers with oil quality and insulation monitoring
- **Power Distribution**: Motors with vibration analysis and bearing diagnostics
- **Power Quality**: Capacitor banks with harmonic distortion detection
- **Backup Systems**: UPS with battery health and charger efficiency tracking

### **3. Advanced Technical Implementation**
- **Beyond basic prediction** to actionable prescriptive analytics
- **Deep learning approach** for complex temporal pattern recognition
- **Class imbalance handling** for rare but critical failure events
- **Reinforcement learning** for optimal maintenance scheduling
- **Business value quantification** with clear ROI calculations

### **4. SPE Boot Camp Alignment**
Directly addresses Dr. Luigi Saputelli's objectives:
- Moves from **descriptive to prescriptive analytics**
- Tackles **midstream operational challenges** effectively
- Demonstrates **practical AI/ML application** in energy sector
- Enables **data-driven maintenance decisions**

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
xgboost>=1.4.0
imbalanced-learn>=0.8.0
```

## 📄 Documentation

Complete project documentation is available in [Final_Project_Report.md](Final_Project_Report.md), including:

### **Comprehensive Coverage**
- **Problem Definition**: Real challenges from Bapco Gas internship
- **Data Sources**: Synthetic data generation based on hands-on experience
- **Advanced AI/ML Methods**: LSTM autoencoders, SMOTE, hyperparameter tuning
- **Performance Analysis**: Detailed metrics with 92-95% accuracy achievements
- **Results Architecture**: Seven-category organization of 200+ analytical outputs
- **Business Impact**: Prescriptive recommendations with quantified ROI
- **Implementation Roadmap**: Short/medium/long-term deployment strategy

### **Professional Standards**
- Industry-standard documentation format
- Technical depth with business accessibility
- Complete submission package for SPE Boot Camp
- Ready for professional presentation and showcase

## 🏆 Submission Details

### **SPE AI/ML Boot Camp Final Project**
- **Course**: SPE AI Machine Learning for Energy Professionals Boot Camp  
- **Instructor**: Dr. Luigi Saputelli  
- **Submission Deadline**: August 18, 2025, 11:59 PM GST  
- **Contact**: abdulla.moosa@bapcoenergies.com  
- **Recognition**: Top projects showcased at SPE Bahrain Local section meeting (August 2025)
- **Awards**: Prizes for 1st, 2nd, and 3rd place projects

### **Deliverables Package**
1. **Complete GitHub repository** with full implementation
2. **Final_Project_Report.md** with comprehensive documentation
3. **Working code** (data generation, model training, prescriptive engine)
4. **Jupyter notebooks** for analysis and demonstration
5. **Results visualizations** across seven analytical categories
6. **Professional README** with quick start guide

## 🤝 Contributing

This project was developed as a final submission for the SPE Boot Camp. For reuse or adaptation:

1. **Credit the original work**: Reference author and SPE Boot Camp
2. **Maintain educational focus**: Preserve industry learning objectives
3. **Consider appropriate licensing**: For broader commercial use
4. **Extend responsibly**: Build upon the prescriptive analytics foundation

## 📞 Contact

**Author**: Jaffar Hasan  
**Role**: Electrical Engineering Intern, Maintenance Department, Bapco Gas  
**Submission Contact**: jaffarhassan2003@gmail.com, +973 3734 1074  
**Project Context**: SPE AI/ML Boot Camp Final Project

---

## 🌟 Project Impact

*This project demonstrates the practical application of advanced AI/ML techniques to solve real-world maintenance challenges in the energy industry. By transitioning from reactive to prescriptive maintenance practices, it embodies the SPE Boot Camp's vision of enabling energy professionals to tackle operational problems effectively using cutting-edge analytics and machine learning tools.*

### **Key Achievements**
✅ **Authentic Industry Application**: Real Bapco Gas maintenance experience  
✅ **Advanced AI/ML Stack**: LSTM, SMOTE, Bayesian optimization, reinforcement learning  
✅ **Prescriptive Analytics**: Beyond prediction to actionable recommendations  
✅ **Business Value Focus**: Quantified ROI and operational impact  
✅ **Professional Documentation**: Industry-standard submission package  
✅ **SPE Boot Camp Excellence**: Ready for showcase and competition  

---

*Ready for submission to SPE AI/ML Boot Camp - August 18, 2025* 🚀
