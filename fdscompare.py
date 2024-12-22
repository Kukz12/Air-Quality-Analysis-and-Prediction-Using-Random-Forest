# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix

# Load the dataset
file_path = "C:/Users/sneha/OneDrive/Documents/city_day.csv"
df = pd.read_csv(file_path)

# Step 1: Preprocessing
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Select relevant columns (pollutants + AQI)
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Handle missing values
df[pollutants] = df[pollutants].apply(pd.to_numeric, errors='coerce')
df[pollutants] = df[pollutants].fillna(df[pollutants].mean())
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df['AQI'] = df['AQI'].fillna(df.groupby('City')['AQI'].transform('mean'))

# Filter data for a specific city
selected_city = "Delhi"  # Change this for other cities
city_data = df[df['City'] == selected_city].dropna()

# Step 2: Correlation Analysis
# Calculate the correlation matrix
corr_matrix = city_data[pollutants + ['AQI']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Correlation Matrix for {selected_city}")
plt.show()

# Feature Selection
# Identify top features correlated with AQI
best_features = corr_matrix['AQI'].sort_values(ascending=False).index[1:6]  # Top 5 features
print("Best Features for Predicting AQI:", list(best_features))

# Prepare data
X = city_data[best_features]
y = city_data['AQI']

# Classification: Categorize AQI into ranges
bins = [0, 50, 100, 150, 200, 300, np.inf]
labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
y_class = pd.cut(y, bins=bins, labels=labels)

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification Model
# Training a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_class_train)

# Predict and evaluate
y_class_pred = classifier.predict(X_test)

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))
accuracy_class = accuracy_score(y_class_test, y_class_pred)
print("Classification Accuracy:", accuracy_class)

# Visualize confusion matrix
conf_matrix = confusion_matrix(y_class_test, y_class_pred, labels=labels)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Regression Model
# Train a Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
error_percentage = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\nRegression Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Error Percentage:", error_percentage)

# Visualize Actual vs Predicted AQI
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title("Actual vs Predicted AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.show()

# Step 6: Next-Day Prediction
# Predict AQI for the next day
next_day_input = city_data[best_features].iloc[-1:].values  # Last available row
next_day_class = classifier.predict(next_day_input)
next_day_reg = regressor.predict(next_day_input)

print("\nNext Day Predictions:")
print(f"Classification Prediction (Range): {next_day_class[0]}")
print(f"Regression Prediction (Exact Value): {next_day_reg[0]:.2f}")

