# Movie Genre Prediction

This project aims to predict the genre of movies based on their descriptions using BERT and RoBERTa models. The dataset used for training and evaluation contains movie overviews and their corresponding genres.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
## Project Overview

Movie genre prediction is a multi-label classification task where each movie description is associated with one or more genres. In this project, we utilize BERT and RoBERTa models to perform this classification task.

## Dataset

The dataset used in this project is sourced from [Kaggle's Movie Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies). It contains movie titles, overviews, and genres. The data has been preprocessed to remove missing values and only includes the necessary columns.
# rename the dataset as movies.csv!!
The dataset file used in this project is `preprocessed_movies.csv`, which includes the following columns:
- `title`: The title of the movie
- `overview`: The description of the movie
- `genres`: The genres of the movie

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```
## Training
To train the models, you can use the provided training scripts. Here are the steps:

1. BERT Model
```bash
python BERT_train.py
```
2. RoBERTa Model
bash
```bash
python RoBERTa_train.py
```

These scripts will train the respective models on the movie dataset and save the trained models in the saved_model directory. The training scripts include progress bars and logging for monitoring the training process.

## Prediction
After training the models, you can use the prediction scripts to classify new movie descriptions. Here are the steps:

1. BERT Model Prediction
```bash
python BERT_predict.py
```
2. RoBERTa Model Prediction
```bash
python RoBERTa_predict.py
```

The prediction scripts will load the trained models and tokenizer, take a movie description as input, and output the predicted genres along with their probabilities.

## Results
The results of the model training and evaluation, including the loss and accuracy curves, are saved and visualized in the training scripts. The evaluation metrics (accuracy, precision, recall, F1 score) are saved in metrics.json.

## Acknowledgements
The dataset used in this project is sourced from Kaggle.
The models are implemented using the Hugging Face Transformers library.
Special thanks to the developers and contributors of the libraries and tools used in this project.
markdown
Copy code

### Explanation

1. **Project Overview**:
   - Provides a brief description of the project and its objectives.

2. **Dataset**:
   - Details the source and structure of the dataset used in the project.

3. **Requirements**:
   - Lists the required Python packages and provides a command to install them using `requirements.txt`.

4. **Training**:
   - Explains how to train the models using the provided training scripts.

5. **Prediction**:
   - Provides instructions on how to use the trained models to predict movie genres for new descriptions.

6. **Results**:
   - Describes where the results of the training and evaluation are saved and how they are visualized.

7. **Acknowledgements**:
   - Gives credit to the data source and the libraries used in the project.





