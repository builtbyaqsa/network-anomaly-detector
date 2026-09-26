# Network Anomaly & Intrusion Detection System 🛡️

An end-to-end Machine Learning pipeline engineered to detect and classify network security threats using the NSL-KDD benchmark dataset. Built with a modular architecture, this project incorporates statistical feature processing, Random Forest classification, model performance evaluation, and a real-time inference wrapper.

---

## 📌 Project Architecture & Highlights

* **Data Engineering & Preprocessing:** Parsed multi-feature network logs, cleaned noise, mapped multiclass traffic indicators to binary outcomes (`Normal: 0`, `Attack: 1`), and executed label encoding for categorical protocols (`tcp`, `udp`, `icmp`).
* **Feature Scaling & Selection:** Utilized `StandardScaler` to handle variance across packet length/duration metrics and applied tree-based feature importance ranking to identify critical intrusion indicators.
* **Model Pipeline & Artifact Serialization:** Trained Decision Tree and Random Forest models, persisting optimal binary model weights (`random_forest_model.pkl`) and feature scalers (`scaler.pkl`) via `joblib`.
* **Inference Pipeline:** Designed a modular script (`day6_inference.py`) capable of parsing real-time packet feature vectors to return classification outcomes along with confidence probabilities.

---

## 📁 Repository Structure

```text
network-anomaly-detector/
├── data/                  # Dataset placeholder (git-ignored)
├── models/                # Serialized model and scaler artifacts (git-ignored)
├── src/
│   ├── day1_exploration.py       # Dataset analysis & target distribution
│   ├── day2_preprocessing.py     # Label encoding & binary target creation
│   ├── day3_scaling_selection.py # StandardScaler & feature importance
│   ├── day4_model_training.py    # Classifier training & model serialization
│   ├── day5_model_evaluation.py  # Precision, Recall, F1-Score & Confusion Matrix
│   └── day6_inference.py         # Production inference wrapper script
├── .gitignore             # Excludes raw data, binaries, and local virtualenv
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies 
 
 ## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Core Machine Learning:** Scikit-learn, Pandas, NumPy
* **Artifact Serialization:** Joblib
* **Environment & Version Control:** Git, GitHub

---

## 🚀 Getting Started

### 1. Clone the Repository & Setup Environment

```cmd
git clone https://github.com/builtbyaqsa/network-anomaly-detector.git
cd network-anomaly-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt 
```cmd

###2. Run the Full ML Pipeline
Execute the modules sequentially to reproduce data preprocessing, training, and evaluation:
python src/day1_exploration.py
python src/day2_preprocessing.py
python src/day3_scaling_selection.py
python src/day4_model_training.py
python src/day5_model_evaluation.py 

###3. Real-Time Packet Inference
Test the trained model with sample network vector inputs:
python src/day6_inference.py