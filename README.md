# QR Code Authentication System

## 📋 Project Description
Machine learning system to detect counterfeit QR codes by analyzing print artifacts and Copy Detection Patterns (CDPs). Achieves 100% accuracy with Random Forest classifier.
(assignment)
## 🛠 Installation
```bash
git clone https://github.com/jhanaviii/QR_Code.git
cd QR_Code
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

🏗 Directory Structure
Copy
QR_Code/
├── data/
│   ├── First Print/    # Original QR codes
│   └── Second Print/   # Counterfeit QR codes
├── models/             # Saved models
├── results/            # Evaluation outputs
├── utils/              # Utility functions
└── main.py             # Main execution script
🚀 Usage
bash
Copy
# Train and evaluate both models
python main.py

# Custom training
from train_ml import train_and_save_ml_model
train_and_save_ml_model(n_estimators=300)
📊 Results
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	1.00	1.00	1.00	1.00
CNN	0.88	0.89	0.88	0.88
📜 License
MIT License

