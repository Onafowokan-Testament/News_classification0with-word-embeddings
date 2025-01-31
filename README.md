# Text Classification with Machine Learning 📄

Welcome to the **Text Classification** project! This project demonstrates how to classify text data into predefined categories using machine learning models. The dataset consists of news stories categorized into sections, and the goal is to predict the correct section for a given story. The project uses **spaCy** for text preprocessing and **scikit-learn** for building and evaluating machine learning models.

---

## Features

- **Text Preprocessing**: Uses spaCy for tokenization, lemmatization, and stopword removal.
- **Feature Extraction**: Converts text into numerical vectors using spaCy's word embeddings.
- **Model Training**: Trains multiple machine learning models, including Decision Trees, Naive Bayes, K-Nearest Neighbors, Random Forests, and Gradient Boosting.
- **Model Evaluation**: Evaluates models using classification reports and confusion matrices.

---

## How It Works

1. **Data Loading**: Load the dataset containing news stories and their corresponding sections.
2. **Text Preprocessing**: Clean and preprocess the text data using spaCy.
3. **Feature Extraction**: Convert the preprocessed text into numerical vectors using spaCy's word embeddings.
4. **Model Training**: Train multiple machine learning models on the vectorized data.
5. **Model Evaluation**: Evaluate the models using classification reports and confusion matrices.

---

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pandas
- numpy
- scikit-learn
- spaCy
- matplotlib
- seaborn

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/text-classification.git
   cd text-classification
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_md
   ```

4. **Run the script**:
   ```bash
   python text_classification.py
   ```

---

## Results

### Model Performance

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Decision Tree          | 0.80     | 0.79      | 0.79   | 0.79     |
| Naive Bayes            | 0.80     | 0.82      | 0.80   | 0.79     |
| K-Nearest Neighbors    | 0.90     | 0.91      | 0.90   | 0.90     |
| Random Forest          | 0.93     | 0.93      | 0.93   | 0.93     |
| Gradient Boosting      | 0.92     | 0.92      | 0.92   | 0.92     |

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's predictions versus the actual labels. It is visualized using a heatmap for better interpretation.

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [spaCy](https://spacy.io/) for text preprocessing and word embeddings.
- [scikit-learn](https://scikit-learn.org/) for machine learning models and evaluation.
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for data visualization.

---

Classify text data with ease using this Text Classification project! 🚀📊
