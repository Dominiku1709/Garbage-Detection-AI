# Garbage Detection AI  
### Group 4 – Tháp tư 119

## 🧭 Introduction
Garbage Detection AI is a deep-learning project designed to automatically detect and classify waste using computer-vision models such as **YOLO** and **RT-DETR**.  
The system supports:

- Real-time webcam detection  
- Image-based detection  
- Object tracking  
- Automatic saving of detection results (cropped images + metadata)  

This project aims to help automate waste sorting and improve environmental management using AI.

---

## 📁 Project Structure


Project structure
```
project_root/
│
├── main.py # Streamlit entry point
│
├── app/
│ ├── run_image_app.py # Image detection interface
│ ├── run_realtime_app.py # Webcam detection & tracking
│
├── inference/
│ ├── detector.py # Unified YOLO + RT-DETR inference
│ ├── tracking.py # Real-time tracking system
│
├── models/
│ ├── loader.py # Load all models
│ ├── rtdetr_loader.py # RT-DETR config loader
│
├── utils/
│ ├── output_saver.py # Save crops + metadata
│ ├── draw.py # Box drawing utilities
│ ├── class_remap.py # Fine → 6 class remapping
│ ├── utils.py # Helper functions
│
└── output/ # Auto-generated detection sessions
```

---

### **Notice**

Create an output folder at project root named "output" before inference run. (If there's none)


## ▶️ How to Run Inference

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **2. Run the app**
```
streamlit run main.py
```

### 