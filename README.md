# 🤖 AutoML No-Code Machine Learning Platform

A Streamlit-based **No-Code AutoML Platform** that allows users to upload datasets, preprocess data, train multiple machine learning models automatically, and select the best-performing model for **classification or regression tasks**.

This project is designed for **final-year academic submission** and supports both **small and large datasets** intelligently.

---

## ✨ Features

- Upload CSV datasets
- Automatic preprocessing (encoding, scaling)
- Supports Classification & Regression
- Handles **small datasets** using SMOTE & data augmentation
- Trains **multiple ML models**
- Automatically selects the **best model**
- Interactive Streamlit UI
- Works on **macOS & Windows**


---

## 🖥️ System Requirements

- Python 3.9 or above
- pip (Python package manager)
- Git (optional, for cloning repository)

---

# ⚙️ Installation Guide (Copy–Paste Ready)

This section explains how to install and run the project on **macOS** and **Windows** step by step.

---

## 📌 Prerequisites

Before starting, make sure you have:
- Python **3.9 or higher**
- pip (comes with Python)
- Git (optional but recommended)

---

# 🟢 Installation on macOS

### Step 1: Check Python Version
```bash
python3 --version
```

## If Python is not installed:
```bash
brew install python
```
### Step 2: Clone the Repository
```bash
git clone  https://github.com/Bhupanimounika22/automl.git
cd automl_app
```

### Step 3: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 4: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 5: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 6: Run the Application
```bash
streamlit run app.py
```


#### The application will open automatically in your default web browser.

# 🟦 Installation on Windows
### Step 1: Check Python Version
```bash
python --version
```

### If Python is not installed:

Download from: https://www.python.org/downloads/

During installation, check “Add Python to PATH”

### Step 2: Clone the Repository
```bash
git clone https://github.com/Bhupanimounika22/automl.git
cd automl_app
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment
```bash
venv\Scripts\activate
```

### Step 5: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 6: Run the Application
```bash
streamlit run app.py

```
