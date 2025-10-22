# 🫁 Pneumonia Detection Website  
### (Django + TensorFlow + VGG19)

A **Deep Learning–powered** web application built using **Django**, **TensorFlow**, and **VGG19** to detect **Pneumonia** from chest X-ray images.  
Supports both **single** and **multiple image** predictions with an intuitive, responsive interface.

---

## 🧠 Tech Stack

| Layer | Technology |
|:------|:------------|
| 🎨 **Frontend** | HTML, CSS, JavaScript |
| ⚙️ **Backend** | Django (Python) |
| 🧩 **Model** | VGG19 (Pretrained on ImageNet, fine-tuned for Pneumonia detection) |
| ☁️ **Deployment** | Render / Any cloud platform |

---

## 🚀 Features

✅ Deep Learning model (**VGG19**) for accurate Pneumonia detection  
🖼️ Upload **single** or **multiple** chest X-ray images  
📊 **Real-time predictions** — Normal or Pneumonia  
🌐 Easy deployment on Render or any cloud provider  
🔒 Secure, scalable, and modular Django structure  
⚡ Integrated TensorFlow 2.x backend  

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Pneumonia-Detection-Django.git
   cd Pneumonia-Detection-Django

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run migrations:
   ```bash
   python manage.py migrate

5. Start the server:
   ```bash
   python manage.py runserver

---

## 💀 My Virtual Environment Struggle

While making this project I fucked up the virtual environment thing!!!

I made this project first without initializing the virtual environment and lost track of some small dependencies
(which I don't know among the shit load of global dependencies).

While trying to correct the mistake, I installed all the major dependencies in venv but the problem

### "List object has no attribute 'shape'" kept arising.
### I mean wtf does that even mean??!!

Then I had to trial and error some global dependencies to find the correct ones, but nope — still the same error.
Got frustrated and finally pip freezed all the global dependencies into requirements.txt.

Now everything works... somehow.

---

## 🧩 Major Dependencies

| 📦 Package | 💡 Description |
|:-----------|:---------------|
| 🟦 `Django` | High-level Python web framework for rapid development |
| 🟩 `TensorFlow` | Deep Learning library used to load and run the pneumonia detection model |
| 🟨 `Keras` | Simplified API for building and deploying neural networks |
| 🟪 `NumPy` | Numerical computing library for handling array operations |
| 🟧 `Pandas` | Data manipulation and analysis library |
| 🩵 `OpenCV-Python` | Image processing and computer vision toolkit |
| 🟥 `Matplotlib` | Visualization library used for displaying image outputs |
| 🟫 `Pillow` | Image handling and conversion library |
| ⚫ `Gunicorn` | WSGI HTTP server for deploying Django applications |
| ⚪ `WhiteNoise` | Simplifies static file serving in production |
| 🟩 `python-dotenv` | Loads environment variables from `.env` files |


---

## Environmental Setup
- Create a .env file in the root directory with:
  ```bash
  SECRET_KEY=your_random_secret_key
  DEBUG=True

---

## 📸 Screenshots

### 🖼️ Single Image Detection
<img width="1920" height="942" alt="Screenshot 2025-10-22 at 17-13-44 Pneumonia Detection" src="https://github.com/user-attachments/assets/f5abf5e6-cabe-4d98-87d9-e136f6191071" />

### 🧮 Multiple Image Detection
<img width="1920" height="881" alt="screenshot-127 0 0 1_8000-2025 10 22-17_14_36" src="https://github.com/user-attachments/assets/eb8848ab-6a44-4aac-989b-119b715e0321" />


