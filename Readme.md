# Intent Studio — Banking77 Intent Classifier

A Streamlit demo that classifies a banking-related customer message into
one of the intent labels known by a trained artificial neural network (ANN).

## Design & Acknowledgements

The model, preprocessing pipeline, and machine learning workflow were
developed by me. The Streamlit UI (`ui.py`) was created with the
assistance of AI, as UI/visual design isn't my area of expertise.

My focus for this project was on building and understanding the
machine learning model rather than designing the frontend from scratch.

## Features

- Dark, responsive Streamlit interface.
- Predicts an intent from a text message.
- Shows the top five model scores.
- Lists intent classes from the saved label encoder.
- Includes an in-app README.
- Checks model/vectorizer feature counts and model/encoder class counts.

## Project structure

```text
project/
├── ui.py
├── README.md
└── banking77_model/
    ├── ann.keras
    ├── tfidf.pkl
    └── label_encoder.pkl