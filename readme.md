# Phishing Email Detector

This project implements a phishing email detection model using a Bidirectional Long Short-Term Memory (BiLSTM) network and binary cross-entropy loss function. The model is designed to classify emails as either legitimate or phishing.

## Table of Contents

- Project Overview
- Features
- Dataset
- Model Architecture
- Installation
- Usage
- Results
- Future Work

### Project Overview
Phishing attacks are one of the most prevalent forms of cybercrime. This project aims to mitigate such threats by building a machine learning model that effectively identifies phishing emails. The model leverages a Bidirectional LSTM architecture to capture the sequential dependencies in text data.

### Features
- BiLSTM Architecture: Utilizes Bidirectional LSTMs for better context understanding from email text.
- Binary Cross-Entropy Loss: Ensures robust optimization for binary classification.
- Preprocessing Pipelines: Includes text cleaning, tokenization, and padding.
- Metrics: Provides accuracy, precision, recall, F1-score, and confusion matrix for evaluation.

### Dataset
- Source: A public dataset containing labeled phishing and legitimate emails. [Click here](https://github.com/Sabal-Subedi/Phishing_Email_detection/tree/main/images/data.png?raw=true)

### Preprocessing:
- Tokenization using Keras Tokenizer.
- Padding sequences to ensure uniform input size.
- Removal of stop words and stemming using NLTK.

### Model Architecture
- Embedding Layer: Converts text data into dense vector representations.
- Bidirectional LSTM Layer: Captures sequential dependencies in both forward and backward directions.
- Dropout Layer: Prevents overfitting by randomly dropping units during training.
- Dense Layer: Outputs the final probability for classification.

### Loss Function
- Binary cross-entropy is used for loss computation, suitable for binary classification tasks.

### Installation
- Prerequisites
  - Python 3.8+

### Dependencies
- Install required dependencies using:
- pip install -r requirements.txt
- Clone the Repository
  - git clone https://github.com/Sabal-Subedi/Phishing_Email_detection

### Usage
- Prepare the Dataset
- Place the dataset in the data/ directory.
- Update the dataset path in the configuration file.
- Train the Model
- Evaluate the Model
- Visualize Results

### Results
The model achieved high accuracy in detecting phishing emails.

- [Loss curve here](https://github.com/Sabal-Subedi/Phishing_Email_detection/tree/main/images/loss.png?raw=true)
- [Accuracy curve here](https://github.com/Sabal-Subedi/Phishing_Email_detection/tree/main/images/accuracy.png?raw=true)
- [Confusion matrix here](https://github.com/Sabal-Subedi/Phishing_Email_detection/tree/main/images/confuse.png?raw=true)
- [Recall here](https://github.com/Sabal-Subedi/Phishing_Email_detection/tree/main/images/recall.png?raw=true)
Evaluation metrics such as precision, recall, and F1-score are reported in the logs.

### Future Work
- Experiment with advanced architectures like Transformers or BERT.
- Extend the model to classify multiple email categories.
- Deploy the model as a real-time email filter.
