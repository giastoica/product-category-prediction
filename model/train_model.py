import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import re
import os
import warnings

# Suppress the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")

# 1. Data Loading and Cleaning
url = "https://raw.githubusercontent.com/giastoica/product-category-prediction/main/data/products.csv"
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error loading data: {e}")

# Fix column names to remove any leading/trailing whitespace
df.columns = df.columns.str.strip()

# Drop rows where 'Product Title' is missing, as it's the core feature
df_clean = df.dropna(subset=["Product Title"]).copy()

# Split the data into a labeled set for training and an unlabeled set for final predictions
df_train = df_clean[df_clean["Category Label"].notna()].copy()

# Normalize 'Category Label' to lowercase
df_train.loc[:, "Category Label"] = df_train["Category Label"].str.lower()

# Impute missing numeric values with the median
for col in ["Number_of_Views", "Merchant Rating"]:
    median_value = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_value)

# 2. Feature Engineering
def has_special_chars(text):
    return 1 if re.search(r'[!@#$%^&*()_+={}\[\]:;"\'<,>/?\\|`~]', text) else 0

def has_all_caps_word(text):
    return 1 if any(word.isupper() and len(word) > 1 for word in text.split()) else 0

brands = ['Acer', 'Apple', 'Dell', 'HP', 'Lenovo', 'Samsung', 'Sony', 'LG', 'Microsoft', 'Nikon']

# New function to check for fridge freezer keywords and product codes
def is_fridge_freezer(title):
    title_lower = str(title).lower()
    keywords = ['fridge freezer', 'side by side', 'freezer on top', 'bottom freezer', 'combi']
    # You can expand this list of product codes based on your data analysis
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
    
    df_copy['Listing Date'] = pd.to_datetime(df_copy['Listing Date'], errors='coerce')
    df_copy['Listing Month'] = df_copy['Listing Date'].dt.month
    df_copy['Listing Day of Week'] = df_copy['Listing Date'].dt.dayofweek
    
    return df_copy.drop(columns=['Listing Date'])

df_train_enriched = feature_engineer(df_train)

# 3. Model Training on Entire Dataset
text_feature = "Product Title"
numeric_features = [
    "Title Word Count", "Title Char Count", "Has Numbers", "Has Special Chars",
    "Has All Caps Word", "Longest Word Length", "Is Brand Mentioned",
    "Listing Month", "Listing Day of Week", "Is_Fridge_Freezer" # <-- Added the new feature here
]
all_features = [text_feature] + numeric_features

X = df_train_enriched[all_features]
y = df_train_enriched["Category Label"]

# Encode the text labels into numerical format for the model
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the final preprocessor and model
preprocessor = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(stop_words="english", max_features=5000), text_feature),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]), numeric_features)
    ]
)

final_classifier = LinearSVC(class_weight='balanced')

final_model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", final_classifier)
])

print("Training final model on the entire labeled dataset...")
final_model_pipeline.fit(X, y_encoded)
print("Final model training complete.")

# 4. Save the Model and Label Encoder using Joblib
model_filename = "final_product_category_model.joblib"
joblib.dump(final_model_pipeline, model_filename)
print(f"\n✅ Final model pipeline saved to '{model_filename}'")

# Save the label encoder separately
encoder_filename = "label_encoder.joblib"
joblib.dump(label_encoder, encoder_filename)
print(f"✅ Label encoder saved to '{encoder_filename}'")