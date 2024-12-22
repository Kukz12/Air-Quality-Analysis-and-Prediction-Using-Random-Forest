# Air-Quality-Analysis-and-Prediction-Using-Random-Forest
**Air Quality Analysis and Prediction Using Random Forest** leverages the `city_day.csv` dataset to analyze pollution trends and predict AQI through classification and regression models. It includes preprocessing, feature selection, visualizations, and next-day AQI predictions, showcasing machine learning applications in environmental science.
Air Quality Analysis and Prediction Using Random Forest

This repository provides a comprehensive framework for analyzing air pollution data and predicting the Air Quality Index (AQI) using machine learning. It is built around the city_day.csv dataset and implements preprocessing, feature selection, and predictive modeling to gain insights into air quality trends and forecast future conditions.

Key Features:
Data Preprocessing:

Converts date values into a standard format.
Handles missing values through column-wise mean imputation.
Focuses on key pollutants like PM2.5, PM10, NOx, and others.
Correlation Analysis:

Identifies relationships between pollutants and AQI using a correlation matrix.
Visualizes pollutant impacts through heatmaps to aid feature selection.
Predictive Modeling:

Classification: Uses a Random Forest Classifier to categorize AQI into standard ranges (e.g., Good, Moderate, Hazardous).
Regression: Applies a Random Forest Regressor to predict exact AQI values.
Evaluation Metrics:

Classification metrics include accuracy, confusion matrix, and detailed classification reports.
Regression metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and error percentage.
Visualizations:

Heatmaps for pollutant correlation.
Confusion matrix for classification performance.
Scatter plots comparing actual vs. predicted AQI values.
Next-Day AQI Prediction:

Predicts AQI category and exact value for the next day based on historical data.
Applications:
This project is valuable for environmental researchers, data scientists, and policymakers interested in understanding pollution trends and making data-driven decisions to mitigate air quality issues. It also serves as a learning tool for exploring machine learning techniques in real-world scenarios.

Requirements:
The project uses Python and depends on libraries such as Pandas, NumPy, Seaborn, Matplotlib, and Scikit-learn. Ensure all dependencies are installed before running the code.
