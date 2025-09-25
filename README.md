Product Category Prediction
This project is an end-to-end machine learning pipeline for predicting the product category based on its title and other features. The solution includes data cleaning, feature engineering, model training, and a final interactive script for making real-time predictions.

Table of Contents
Project Goal

Project Structure

Setup and Installation

How to Use

Model Details

Project Goal
The primary objective of this project is to build a robust machine learning model that can accurately classify a product into one of several predefined categories. The model uses a combination of natural language processing (NLP) on the product title and numerical features to make its predictions.

A key feature of this pipeline is its ability to handle common data issues like missing values and inconsistent formatting, as well as its use of manual feature engineering to improve performance on tricky categories like "fridge freezers."

Project Structure
The project is organized into a clean, modular structure for clarity and ease of use.

product-category-prediction/
│
├── data/
│   └── products.csv                     # The raw dataset
│
├── notebooks/
│   └── product_category_prediction.ipynb       # Your Jupyter Notebook for analysis
│
├── model/
│   ├── train_model.py                   # Script to train and save the model
│   ├── predict_category.py              # Script for interactive predictions
│   ├── final_product_category_model.joblib # The saved, trained model
│   └── label_encoder.joblib             # The saved label encoder
│
├── README.md                            # Project documentation (this file)
└── requirements.txt                     # Python dependencies


Setup and Installation
Prerequisites
You must have Python 3.6 or newer installed on your system.



Files
data/products.csv: This file contains the raw product data. The train_model.py script automatically downloads this from a public URL, so you do not need to download it manually.

model/: This folder contains all the necessary scripts and saved model files.

How to Use
The workflow consists of two main steps: training the model and then using it for predictions.

Step 1: Training the Model
To train the model, simply run the train_model.py script. This will download the data, perform all preprocessing and feature engineering, train the model, and save the model and label encoder to the model/ folder.

cd model
python train_model.py


After this script runs successfully, you should see two new files in the model/ directory:

final_product_category_model.joblib

label_encoder.joblib

Step 2: Making Interactive Predictions
Once the model is trained, you can use the predict_category.py script to test it with your own product titles.

cd model
python predict_category.py


The script will prompt you to enter a product title. Type in your title and press Enter to see the predicted category. To exit the program, type exit.

Model Details
The machine learning pipeline is built using scikit-learn and consists of the following components:

Preprocessing: A ColumnTransformer is used to apply different transformations to different columns.

Text Features: TfidfVectorizer is used on the Product Title to convert text into a numerical format that the model can understand.

Numerical Features: SimpleImputer handles missing values, and MinMaxScaler normalizes the numerical features. A key manual feature, Is_Fridge_Freezer, was added to improve classification accuracy for a specific subcategory.

c

The entire process is encapsulated in a Pipeline to ensure that all training and prediction steps are applied consistently.