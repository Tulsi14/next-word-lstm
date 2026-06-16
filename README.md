# 🔮 NextWord — LSTM Next Word Predictor

A deep learning web app that predicts the next word in a sentence using a Long Short-Term Memory (LSTM) neural network trained on Shakespeare's *Hamlet*.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

---

## 📸 Demo

> Type any phrase → https://next-word-lstm-jew8pbbssbpnfr9dhr2jce.streamlit.app/ **Predict Next Word** → see the model's prediction highlighted inline.
**Example:**
```
Input:  "To be or not to"
Output: "To be or not to be"
```

---

## 🧠 How It Works

```
Your phrase  →  Tokenizer  →  LSTM Model  →  argmax  →  Predicted word
```

1. Input text is tokenized into integer sequences
2. Sequence is padded to match the model's expected input length
3. LSTM network outputs a probability distribution over the vocabulary
4. The word with the highest probability (`argmax`) is returned

---

## 🗂️ Project Structure

```
next-word-lstm/
│
├── app.py                  # Streamlit web app
├── next_word_lstm.h5       # Trained LSTM model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── requirements.txt        # Python dependencies
├── .python-version         # Pins Python 3.11 for Streamlit Cloud
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Tulsi14/next-word-lstm.git
cd next-word-lstm
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
numpy
tensorflow-cpu==2.13.0
```

> **Note:** Python 3.11 is required. TensorFlow does not yet support Python 3.12+.

---

## 🏗️ Model Architecture

| Layer | Details |
|---|---|
| Embedding | Maps token indices to dense vectors |
| LSTM (1) | 128 units, returns sequences |
| LSTM (2) | 128 units |
| Dense | Softmax output over full vocabulary |

- **Training corpus:** Shakespeare's *Hamlet*
- **Early stopping** used to prevent overfitting
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub (include `next_word_lstm.h5` and `tokenizer.pkl`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo and set `app.py` as the main file
4. Deploy — done!

> Make sure `.python-version` file with content `3.11` is in the root of your repo.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model training & inference
- **Streamlit** — web app framework
- **NumPy** — numerical operations
- **Pickle** — tokenizer serialization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Built with 🔮 and way too much Shakespeare</p># 🔮 NextWord — LSTM Next Word Predictor

A deep learning web app that predicts the next word in a sentence using a Long Short-Term Memory (LSTM) neural network trained on Shakespeare's *Hamlet*.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

---

## 📸 Demo

> Type any phrase → click **Predict Next Word** → see the model's prediction highlighted inline.

**Example:**
```
Input:  "To be or not to"
Output: "To be or not to be"
```

---

## 🧠 How It Works

```
Your phrase  →  Tokenizer  →  LSTM Model  →  argmax  →  Predicted word
```

1. Input text is tokenized into integer sequences
2. Sequence is padded to match the model's expected input length
3. LSTM network outputs a probability distribution over the vocabulary
4. The word with the highest probability (`argmax`) is returned

---

## 🗂️ Project Structure

```
next-word-lstm/
│
├── app.py                  # Streamlit web app
├── next_word_lstm.h5       # Trained LSTM model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── requirements.txt        # Python dependencies
├── .python-version         # Pins Python 3.11 for Streamlit Cloud
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/next-word-lstm.git
cd next-word-lstm
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
numpy
tensorflow-cpu==2.13.0
```

> **Note:** Python 3.11 is required. TensorFlow does not yet support Python 3.12+.

---

## 🏗️ Model Architecture

| Layer | Details |
|---|---|
| Embedding | Maps token indices to dense vectors |
| LSTM (1) | 128 units, returns sequences |
| LSTM (2) | 128 units |
| Dense | Softmax output over full vocabulary |

- **Training corpus:** Shakespeare's *Hamlet*
- **Early stopping** used to prevent overfitting
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub (include `next_word_lstm.h5` and `tokenizer.pkl`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo and set `app.py` as the main file
4. Deploy — done!

> Make sure `.python-version` file with content `3.11` is in the root of your repo.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model training & inference
- **Streamlit** — web app framework
- **NumPy** — numerical operations
- **Pickle** — tokenizer serialization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Built with 🔮 and way too much Shakespeare</p>
