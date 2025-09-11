# 🩺 Face Mask Detection using Deep Learning

A deep learning project that detects whether a person is wearing a face
mask or not from an image.
Built with TensorFlow/Keras, using VGG16 transfer learning for high
accuracy.

------------------------------------------------------------------------

# 🚀 Features

-   Detects Mask 😷 vs No Mask 🙅‍♂️ in images.
-   Pre-trained on VGG16 for fast and accurate classification.
-   Implemented with Dropout layers to prevent overfitting.
-   Easy-to-use Streamlit app for real-time predictions.

------------------------------------------------------------------------

🛠️ Tech Stack

-   Python 3.10+
-   TensorFlow / Keras
-   OpenCV
-   NumPy & Pandas
-   Streamlit

------------------------------------------------------------------------

📂 Project Structure

    ├── data/                # Dataset (masked & unmasked faces)
    ├── models/              # Saved trained model(s)
    ├── app.py               # Streamlit web app
    ├── train.py             # Model training script
    ├── requirements.txt     # Dependencies
    └── README.md            # Project documentation

------------------------------------------------------------------------

⚙️ Installation

1.  Clone this repository:

        git clone https://github.com/your-username/mask-detection.git
        cd mask-detection

2.  Create and activate a virtual environment (optional but
    recommended):

        python -m venv venv
        source venv/bin/activate   # Linux/Mac
        venv\Scripts\activate      # Windows

3.  Install dependencies:

        pip install -r requirements.txt

------------------------------------------------------------------------

🧑‍💻 Usage

1. Train the model

    python train.py

2. Run the Streamlit app

    streamlit run app.py

Then open your browser at http://localhost:8501 🎉

------------------------------------------------------------------------

📊 Example

  -----------------------------------------------------------------------
  With Mask 😷                    Without Mask 🙅‍♂️
  ------------------------------- ---------------------------------------
  [Mask]                          [No Mask]

  -----------------------------------------------------------------------

------------------------------------------------------------------------

📈 Model Performance

-   Base Model: VGG16 (ImageNet weights, no top layer)
-   Fine-tuned layers: Fully connected layers with Dropout
-   Final Layer: Sigmoid (binary classification)
-   Accuracy: add your accuracy here after training

------------------------------------------------------------------------

🤝 Contributing

Contributions are welcome!
- Fork the repo
- Create a new branch (feature/new-idea)
- Submit a Pull Request

------------------------------------------------------------------------

📜 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

👨‍💻 Author

Developed by Your Name ✨
