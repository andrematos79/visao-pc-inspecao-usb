# SVC – Computer Vision System for USB Connector Inspection

<p align="center">
  <img src="assets/logo_sistema.png" width="450">
</p>

Industrial AI-based computer vision platform for automated detection of **USB connector assembly defects in mobile phone chargers**.

**Inspection Mode:** Single ROI USB Connector Inspection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19323586.svg)](https://doi.org/10.5281/zenodo.19323586)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/badge/version-v0.9--usb-blue)

---

# Overview

**SVC USB (Computer Vision System for USB Inspection)** is a low-cost industrial computer vision platform designed for **automated inspection of USB connectors installed in mobile phone chargers**.

The system uses **deep learning and industrial triggering mechanisms** to detect assembly defects during production.

The platform evolved from the **SVC Spring Inspection architecture**, adapting the same industrial AI framework for USB connector inspection.

---

# Key Features

✔ Low-cost industrial computer vision architecture  
✔ CNN-based inspection using **MobileNetV2**  
✔ Single ROI inspection strategy for USB connector region  
✔ CPU-only inference (no GPU required)  
✔ Automatic inspection triggering via proximity sensor  
✔ Industrial operator interface built with **Streamlit**

---

# System Architecture

Operational pipeline:

Sensor Trigger  
↓  
Image Acquisition (USB Industrial Camera)  
↓  
ROI Extraction (USB Connector Region)  
↓  
CNN Classification (MobileNetV2)  
↓  
Industrial Decision Logic  
↓  
Operator Interface + Production Logging  
↓  
Evidence Storage + Reporting

---

# Hardware Components

Industrial PC (Windows 10 / 11)  
Intel Core i3 12th Gen or higher  
8 GB RAM minimum  
Industrial USB camera  
Arduino Uno microcontroller  
E18-D80NK proximity sensor

---

# Software Stack

Python  
TensorFlow / Keras  
OpenCV  
Streamlit  
PySerial  
Pandas  
Matplotlib

---

# Artificial Intelligence Model

The inspection system uses **MobileNetV2 with Transfer Learning**.

### Model Classes

OK — USB connector correctly assembled  
NG_CORPO_ESTRANHO — Foreign object inside the connector  
NG_DANIFICADO — Damaged connector structure  
NG_DESALINHADO — Connector misaligned during assembly

The CNN analyzes the **USB connector region of interest (ROI)** and classifies the detected condition.

---

# Decision Logic

The system analyzes a **single ROI containing the USB connector**.

The CNN model predicts the connector condition. If a defect is detected, the product is automatically rejected.

This approach allows **fast inspection and reliable detection of assembly defects**.

---

# Dataset Collection System

The SVC USB includes a built-in **dataset generation tool** allowing engineers to capture inspection images directly from production.

Benefits:

• Continuous dataset expansion  
• Faster AI retraining cycles  
• Real industrial defect collection  
• Improved model robustness

Images are automatically organized into structured dataset folders.

---

# Evidence Management System

When a defect is detected, the system automatically stores an **NG evidence image**.

These images support:

• Quality audits  
• Failure investigations  
• Dataset expansion  
• Manufacturing process improvements

Retention options:

30 days  
60 days  
90 days

Older evidence images are automatically removed automatically according to retention policies.

---

# Automated Reporting

The SVC system can generate inspection reports containing:

• Production yield  
• Defect distribution  
• Inspection statistics  
• Traceability data

Reports can be exported for **industrial auditing and quality monitoring**.

---

# Installation

Create project directory:
C:\SVC_INSPECAO_USB

Create virtual environment:
python -m venv .venv_usb


Activate environment:
..venv_usb\Scripts\Activate.ps1


Install dependencies:
pip install -r requirements.txt


---

# Running the System

streamlit run app_camera_infer_usb.py

Or use the launcher:

INICIAR_SVC_USB.bat


---

# Research Context

This project contributes to research in:

• Industrial computer vision  
• Automated quality inspection  
• Deep learning for manufacturing  
• Smart Manufacturing / Industry 4.0
• Control and Automation Engineering


The system demonstrates the feasibility of **deploying deep learning inspection systems using low-cost hardware in real manufacturing environments**.

---


# Citation

If you use this system in research or industrial projects, please cite:

Matos, A. G. (2026)  
**SVC USB – Computer Vision System for USB Connector Inspection**  
Zenodo  
https://doi.org/10.5281/zenodo.19323586

---

## Author

**André Gama de Matos**  
Student — Control and Automation Engineering  

Undergraduate Final Project (TCC)  
Control and Automation Engineering  
Centro Universitário UNIFATECIE

**Advisor**  
Prof.  Lucas Delapria Dias dos Santos

---

# License

MIT License — Open source software for research and industrial experimentation.



