# Livestock Disease Prediction System

This project is a simple web application designed to predict whether livestock is healthy or diseased based on various physiological and categorical features.

## Project Structure

- `app.py`: The main Flask backend containing data preprocessing, model training, and prediction logic.
- `livestock_data.csv`: Synthetic dataset generated for the project.
- `generate_data.py`: Script used to create the synthetic dataset.
- `templates/index.html`: The frontend user interface.
- `static/`: Directory containing generated evaluation graphs.

## Features

1. **Data Loading & Preprocessing**: Loads the CSV, encodes categorical data, and scales numerical values.
2. **Multiple Models**: Implements **Decision Tree**, **Random Forest** (Supervised), and **K-Means Clustering** (Unsupervised).
3. **Evaluation**: Displays performance metrics (Accuracy, Precision, Recall, F1) for supervised models and Cluster Purity/Silhouette Score for unsupervised clustering.
4. **Visualizations**: Includes several analysis graphs (Count plot, Heatmap, Supervised Comparison, ROC Curves, Feature Importance, K-Means Clusters, and Elbow Method).
5. **Prediction**: A functional form where you can input livestock data and get an instant prediction using the Random Forest model.

## How to Run

1. **Install Dependencies**:
   Open your terminal and run:
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the Application**:
   Execute the following command in the project directory:
   ```bash
   python app.py
   ```

3. **Access the Web App**:
   Open your browser and go to: `http://127.0.0.1:5000`

## Developer Notes
- The UI is designed to be simple and functional, as requested.
- The prediction model currently uses Random Forest due to its high reliability in classification tasks.
