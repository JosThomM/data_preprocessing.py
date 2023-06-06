import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        X = data.drop('target_variable', axis=1)
        y = data['target_variable']

        # Perform label encoding on categorical variables
        categorical_cols = X.select_dtypes(include='object').columns
        X[categorical_cols] = X[categorical_cols].apply(self.label_encoder.fit_transform)

        # Perform feature scaling on numerical variables
        numerical_cols = X.select_dtypes(include=['int', 'float']).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        return X, y

# Example usage:
file_path = 'path/to/your/data.csv'

preprocessor = DataPreprocessor()

data = preprocessor.load_data(file_path)
X, y = preprocessor.preprocess_data(data)

print("Preprocessed X:")
print(X.head())
print("Target variable:")
print(y.head())
