import joblib
import pandas as pd
import re
import os

# 1. Load the Saved Model and Encoder
model_filename = "final_product_category_model.joblib"
encoder_filename = "label_encoder.joblib"

if not os.path.exists(model_filename) or not os.path.exists(encoder_filename):
    print(f"Error: One or both model files ('{model_filename}' and '{encoder_filename}') were not found.")
    print("Please make sure you have run the 'train_model.py' script first.")
    exit()

loaded_model = joblib.load(model_filename)
loaded_encoder = joblib.load(encoder_filename)

print("✅ Model and encoder loaded successfully. Ready for interactive predictions.")
print("Type 'exit' to quit the program.")

# 2. Define the Feature Engineering Functions (must be the same as training)
def has_special_chars(text):
    return 1 if re.search(r'[!@#$%^&*()_+={}\[\]:;"\'<,>/?\\|`~]', text) else 0

def has_all_caps_word(text):
    return 1 if any(word.isupper() and len(word) > 1 for word in text.split()) else 0

brands = ['Acer', 'Apple', 'Dell', 'HP', 'Lenovo', 'Samsung', 'Sony', 'LG', 'Microsoft', 'Nikon']

# New function to check for fridge freezer keywords and product codes
def is_fridge_freezer(title):
    title_lower = str(title).lower()
    keywords = ['fridge freezer', 'side by side', 'freezer on top', 'bottom freezer', 'combi']
    product_codes = ['sbs8004po', 'sbs96312', 'nf8300', 'nf68321']

    if any(k in title_lower for k in keywords):
        return 1
    if any(pc in title_lower for pc in product_codes):
        return 1

    return 0

def feature_engineer(df):
    df_copy = df.copy()
    df_copy['Title Word Count'] = df_copy['Product Title'].apply(lambda x: len(x.split()))
    df_copy['Title Char Count'] = df_copy['Product Title'].apply(len)
    df_copy['Has Numbers'] = df_copy['Product Title'].apply(lambda x: int(any(char.isdigit() for char in x)))
    df_copy['Has Special Chars'] = df_copy['Product Title'].apply(has_special_chars)
    df_copy['Has All Caps Word'] = df_copy['Product Title'].apply(has_all_caps_word)
    df_copy['Longest Word Length'] = df_copy['Product Title'].apply(lambda x: max((len(word) for word in x.split()), default=0))
    df_copy['Is Brand Mentioned'] = df_copy['Product Title'].apply(lambda x: int(any(brand.lower() in x.lower() for brand in brands)))
    
    # Add the new feature here
    df_copy['Is_Fridge_Freezer'] = df_copy['Product Title'].apply(is_fridge_freezer)
    
    # Use dummy values for non-text features since the user can't input them
    df_copy['Listing Month'] = 1  
    df_copy['Listing Day of Week'] = 1  
    
    return df_copy

# 3. Interactive Loop for Predictions
while True:
    user_input = input("\nEnter a product title (or 'exit' to quit): ").strip()
    
    if user_input.lower() == 'exit':
        print("Exiting interactive predictor. Goodbye!")
        break
    
    if not user_input:
        print("Please enter a valid product title.")
        continue
    
    # Create a DataFrame with the necessary columns for the feature engineering function
    input_data = pd.DataFrame([{
        "Product Title": user_input,
        "Number_of_Views": 0.0,
        "Merchant Rating": 0.0,
        "Listing Date": "01/01/2024"
    }])
    
    # Engineer features from the user's input
    enriched_input = feature_engineer(input_data)
    
    # Define the list of features needed by the model
    all_features = [
        "Product Title", "Title Word Count", "Title Char Count", "Has Numbers",
        "Has Special Chars", "Has All Caps Word", "Longest Word Length",
        "Is Brand Mentioned", "Listing Month", "Listing Day of Week", "Is_Fridge_Freezer"
    ]
    
    # Make a prediction
    try:
        prediction_encoded = loaded_model.predict(enriched_input[all_features])
        prediction_label = loaded_encoder.inverse_transform(prediction_encoded)
        print(f"Predicted Category: {prediction_label[0].upper()}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        print("Please ensure your input is valid and the model is correctly loaded.")