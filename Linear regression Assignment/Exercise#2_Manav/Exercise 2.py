# Part a: Get the data
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
path = r'C:\Users\manav\PycharmProjects\Linear Regression assignment\Exercise 2'
filename = 'BikeSharingData.csv'
fullpath = os.path.join(path, filename)
bikesharing_Manav = pd.read_csv(fullpath)


# Part b: Initial Exploration
print("First 5 records:")
print(bikesharing_Manav.head(5))

print("\nColumn names:")
print(bikesharing_Manav.columns.values)

print("\nDataframe shape:")
print(bikesharing_Manav.shape)

print("\nColumn types:")
print(bikesharing_Manav.dtypes)

print("\nMissing values:")
for col in bikesharing_Manav.columns.values:
    missing_count = bikesharing_Manav[col].isnull().sum()
    if missing_count > 0:
        print(f"{col}: {missing_count} missing values")
    else:
        print(f"{col}: No missing values")



# Part c: Data transformation

categorical_columns = ['season', 'holiday', 'weekday', 'workingday', 'weathersit']

bikesharing_Manav = pd.get_dummies(bikesharing_Manav, columns=categorical_columns, drop_first=True)

bikesharing_Manav = bikesharing_Manav.drop(columns=['instant'])

# Normalization function
def normalize_dataframe(dataframe):
    # Select only numeric columns
    numeric_df = dataframe.select_dtypes(include=[np.number])
    return (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())


df_normalized = normalize_dataframe(bikesharing_Manav)

print("\nFirst 5 records after normalization:")
print(df_normalized.head())


# Boxplot
plt.figure(figsize=(10, 9))
df_normalized.boxplot()
plt.xticks(rotation=90)
plt.title("Boxplot of All Variables")
plt.tight_layout()
plt.show()

# Pairwise scatter plot
import seaborn as sns
selected_columns = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
plt.figure(figsize=(12, 10))
sns.pairplot(df_normalized[selected_columns])
plt.suptitle("Pairwise Scatter Plot", y=1.02)
plt.show()

# Part d: Build a model
np.random.seed(84)

# Prepare features and target
features1 = ['temp', 'atemp', 'hum'] + [col for col in df_normalized.columns if col.startswith(('season', 'holiday',
'weekday', 'workingday', 'weathersit'))]

features2 = features1 + ['windspeed']

X1 = df_normalized[features1]
X2 = df_normalized[features2]
y = df_normalized['cnt']

# Split the data
X1_train_Manav, X1_test_Manav, y1_train_Manav, y1_test_Manav = train_test_split(X1, y, test_size=0.2, random_state=84)
X2_train_Manav, X2_test_Manav, y2_train_Manav, y2_test_Manav = train_test_split(X2, y, test_size=0.2, random_state=84)

# Model 1 (without windspeed)
model1 = LinearRegression()
model1.fit(X1_train_Manav, y1_train_Manav)

print("\nModel 1 (without windspeed):")
print("Coefficients:", model1.coef_)
print("R-squared Score:", model1.score(X1_test_Manav, y1_test_Manav))

# Model 2 (with windspeed)
model2 = LinearRegression()
model2.fit(X2_train_Manav, y2_train_Manav)

print("\nModel 2 (with windspeed):")
print("Coefficients:", model2.coef_)
print("R-squared Score:", model2.score(X2_test_Manav, y2_test_Manav))