# 🧠 Brain Tumor Diagnostic & Classification System

An AI-powered web platform designed to assist in multi-class brain tumor classification from MRI scan images using Convolutional Neural Networks (CNN) and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras%203-orange)
![UI](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red)

---

## 🔗 Live Demo
Access the hosted application here: **[Live App Link](https://braintumorproject-ddoa68skhhpefad9paundn.streamlit.app/)**

---

## 📌 Features
- **Multi-Class Tumor Detection:** Classifies input MRI scans into four distinct categories:
  - `Glioma`
  - `Meningioma`
  - `Pituitary`
  - `No Tumor`
- **Real-Time Classification:** Provides primary diagnostic prediction along with confidence percentages.
- **Visual Analytics:** Interactive probability distribution graph rendered via Plotly.
- **Glassmorphic Responsive UI:** Optimized interface built with custom Streamlit styling for mobile & desktop views.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.11
* **Deep Learning Framework:** TensorFlow, Keras 3
* **Data Processing & Math:** NumPy, Pillow (PIL)
* **Frontend & Dashboard:** Streamlit, Plotly
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
Brain_tumorProject/
│
├── app.py                  # Main Streamlit Web Application Script
├── config.json             # Keras CNN Architecture Configuration
├── model.weights.h5        # Trained Model Weights File
├── metadata.json           # Model Metadata Info
├── requirements.txt        # Project Dependencies
├── runtime.txt             # Python Runtime Version for Deployment
└── README.md               # Project Documentation
