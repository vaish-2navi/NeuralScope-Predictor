# 🧠 NeuralScope — ML Prediction Dashboard

A modern dark-mode Machine Learning web application built with Streamlit and a trained Random Forest Classifier. NeuralScope provides an interactive interface for entering PCA-transformed features and viewing model predictions.

## 🚀 Live Demo

**[🌐 Try NeuralScope Live](https://neuralscope-predictor.streamlit.app/)**

## ✨ Features

- 🌙 Modern dark-mode user interface
- 🤖 Random Forest classification model
- 🔢 11 input features (PC1–PC11)
- 🎯 Instant model predictions
- 📊 Class probability visualization
- 🔍 Input feature review
- 🔄 Reset inputs functionality
- ☁️ Deployed using Streamlit Community Cloud

## 🛠️ Tech Stack

- **Language:** Python
- **UI Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Numerical Computing:** NumPy
- **Model Storage:** Pickle
- **Deployment:** Streamlit Community Cloud

## 📂 Project Structure

```text
neural-scope/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── model (1) (1).pkl       # Trained ML model
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## ⚙️ Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🧪 How to Use

1. Open the [live application](https://neuralscope-predictor.streamlit.app/).
2. Enter values for PC1 to PC11.
3. Click **Run prediction**.
4. View the predicted class and class probabilities.
5. Review the submitted feature values if needed.

**Note:** The model expects PCA-transformed input features. Use the same fitted preprocessing pipeline used during training. Interpret class labels using the original dataset documentation.

## 🔮 Future Improvements

- Integrate the complete preprocessing pipeline.
- Add model evaluation metrics.
- Improve input validation and error handling.
- Add prediction history and downloadable results.
- Include model explainability visualizations.

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 👩‍💻 Author

**Vaishnavi Jadhav**

Machine Learning | Artificial Intelligence | Python

---

⭐ If you find this project interesting, consider giving the repository a star!
