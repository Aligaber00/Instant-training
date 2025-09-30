# 🧠 Brain Tumor Detection & Segmentation

This project provides a **Streamlit web app** for brain tumor **detection (classification)** and **segmentation** using deep learning models.

- **Detection**: Predicts whether an MRI scan contains a tumor or not.  
- **Segmentation**: Generates a binary mask highlighting the tumor region.  

---

## 🚀 Features
- Upload an MRI image (`.jpg`, `.jpeg`, `.png`)
- Choose between **Detection** (tumor vs. no tumor) or **Segmentation** (mask output)
- Simple, interactive UI built with **Streamlit**
- Tumor segmentation mask is displayed side-by-side with the uploaded MRI

---

## 📂 Repository Structure
.
├── app.py # Streamlit app
├── brain_tumor.ipynb # Model training / experimentation notebook
├── models/ # (Optional) Pretrained models saved here
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
This will start a local server, usually accessible at:

arduino
Copy code
http://localhost:8501
📝 Example Workflow
Open the app in your browser.

Upload an MRI scan.

Select Detection → see tumor probability and prediction.

Select Segmentation → view the binary mask (white tumor on black background) beside the uploaded scan.

📦 Dependencies
Python 3.8+

Streamlit

TensorFlow / Keras

NumPy

OpenCV

Pillow

(Install automatically via requirements.txt)

📖 Training
The notebook brain_tumor.ipynb contains model training, preprocessing, and evaluation.
You can train your own detection and segmentation models or load pre-trained ones into app.py.


