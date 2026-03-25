from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc, silhouette_score
)
import os
import random

app = Flask(__name__)

# ============================================================
# CREATE STATIC FOLDER — stores all generated graphs/images
# ============================================================
if not os.path.exists('static'):
    os.makedirs('static')


def process_and_test_models():
    """
    Main processing function that:
      1. Loads and samples the dataset
      2. Preprocesses features (encoding, scaling)
      3. Trains SUPERVISED models (Decision Tree, Random Forest)
      4. Runs UNSUPERVISED model (K-Means Clustering)
      5. Generates all comparison graphs and visualizations
    """

    # ============================================================
    # 1. DATA LOADING — Read CSV and sample for performance
    # ============================================================
    print(">> [1/8] Loading dataset...")
    csv_path = 'data set/global_cattle_disease_detection_dataset (1).csv'
    df_full = pd.read_csv(csv_path)
    # Use a sample of 10,000 rows for performance
    df = df_full.sample(n=10000, random_state=42).copy()
    print(f"   Dataset loaded: {len(df_full)} total rows, sampled {len(df)} rows.")

    # ============================================================
    # 2. PREPROCESSING — Encode categories, scale numerics, split
    # ============================================================
    print(">> [2/8] Preprocessing data...")

    # Convert multi-class Disease_Status to binary (Healthy vs Diseased)
    df['Disease_Status'] = df['Disease_Status'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Diseased')

    # Define categorical and numeric feature columns
    categorical_cols = [
        'Breed', 'Region', 'Country', 'Climate_Zone', 'Management_System',
        'Lactation_Stage', 'Feed_Type', 'Season'
    ]
    numeric_cols = [
        'Age_Months', 'Weight_kg', 'Parity', 'Days_in_Milk',
        'Feed_Quantity_kg', 'Water_Intake_L', 'Walking_Distance_km',
        'Grazing_Duration_hrs', 'Rumination_Time_hrs', 'Resting_Hours',
        'Body_Temperature_C', 'Heart_Rate_bpm', 'Respiratory_Rate',
        'Ambient_Temperature_C', 'Humidity_percent', 'Housing_Score',
        'Milk_Yield_L', 'FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine',
        'BQ_Vaccine', 'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine',
        'Rabies_Vaccine', 'Previous_Week_Avg_Yield', 'Body_Condition_Score',
        'Milking_Interval_hrs'
    ]

    selected_features = categorical_cols + numeric_cols
    target = 'Disease_Status'

    # Drop rows with missing values in selected columns
    df = df[selected_features + [target]].dropna()

    # Label-encode all categorical columns
    encoders = {}
    df_encoded = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Label-encode the target variable (Healthy=0 / Diseased=1 or vice-versa)
    le_target = LabelEncoder()
    df_encoded[target] = le_target.fit_transform(df[target])
    encoders[target] = le_target

    # Separate features (X) and target (y)
    X = df_encoded[selected_features]
    y = df_encoded[target]

    # Train-test split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features (used by K-Means)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print(f"   Preprocessing complete. Training set: {len(X_train)}, Test set: {len(X_test)}")

    # ============================================================
    # 3. SUPERVISED MODEL — Decision Tree Classifier
    # ============================================================
    print(">> [3/8] Training Decision Tree (Supervised)...")

    # Initialize and train Decision Tree with default hyperparameters
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Generate predictions on the test set
    dt_pred = dt.predict(X_test)

    # Calculate Decision Tree performance metrics
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_prec = precision_score(y_test, dt_pred)
    dt_rec = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_cm = confusion_matrix(y_test, dt_pred)

    print(f"   Decision Tree — Accuracy: {dt_acc:.4f}, Precision: {dt_prec:.4f}, "
          f"Recall: {dt_rec:.4f}, F1: {dt_f1:.4f}")

    # ============================================================
    # 4. SUPERVISED MODEL — Random Forest Classifier
    # ============================================================
    print(">> [4/8] Training Random Forest (Supervised)...")

    # Initialize and train Random Forest with 100 estimators
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Generate predictions on the test set
    rf_pred = rf.predict(X_test)

    # Calculate Random Forest performance metrics
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)

    print(f"   Random Forest — Accuracy: {rf_acc:.4f}, Precision: {rf_prec:.4f}, "
          f"Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}")

    # ============================================================
    # 5. UNSUPERVISED MODEL — K-Means Clustering
    # ============================================================
    print(">> [5/8] Running K-Means Clustering (Unsupervised)...")

    # Fit K-Means with 2 clusters (Healthy vs Diseased grouping)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Get cluster labels assigned to each sample
    kmeans_labels = kmeans.labels_

    # Calculate the Silhouette Score — measures how well samples fit their cluster
    # Score ranges from -1 (bad) to +1 (good); higher is better
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

    # Calculate cluster inertia — sum of squared distances to nearest cluster center
    kmeans_inertia = kmeans.inertia_

    # Build a cross-tabulation of cluster labels vs actual disease status
    # This shows how well the unsupervised clusters align with true labels
    kmeans_crosstab = pd.crosstab(
        kmeans_labels,
        df_encoded[target],
        rownames=['Cluster'],
        colnames=['Actual']
    )

    # Count how many samples ended up in each cluster
    kmeans_cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index().to_dict()

    # Calculate Cluster Purity — measures how well clusters align with true labels
    # For each cluster, count the majority class; purity = sum(majority counts) / total
    kmeans_purity = sum(kmeans_crosstab.max(axis=1)) / len(kmeans_labels)

    print(f"   K-Means — Silhouette Score: {kmeans_silhouette:.4f}, "
          f"Inertia: {kmeans_inertia:.2f}, Purity: {kmeans_purity:.4f}")
    print(f"   Cluster distribution: {kmeans_cluster_counts}")
    print(f"   Cluster vs Actual:\n{kmeans_crosstab}")

    # ============================================================
    # 6. GRAPHS — Data Overview (Disease distribution & correlations)
    # ============================================================
    print(">> [6/8] Generating data overview graphs...")

    # 6a. Disease Status Distribution — shows class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Disease_Status', data=df, palette='viridis')
    plt.title('Disease Status Distribution (Sampled)')
    plt.savefig('static/countplot.png')
    plt.close()

    # 6b. Correlation Heatmap — shows feature-to-feature correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('static/heatmap.png')
    plt.close()

    # ============================================================
    # 7. GRAPHS — Supervised Model Comparison & Evaluation
    # ============================================================
    print(">> [7/8] Generating supervised model comparison graphs...")

    # 7a. Accuracy Comparison Bar Chart — Decision Tree vs Random Forest
    models = ['Decision Tree', 'Random Forest']
    accuracies = [dt_acc, rf_acc]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=models, y=accuracies, palette='magma')
    plt.title('Supervised Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig('static/model_comparison.png')
    plt.close()

    # 7a-2. All Three Methods Comparison Bar Chart
    # Compares DT Accuracy, RF Accuracy, and K-Means Cluster Purity side by side
    all_models = ['Decision Tree', 'Random Forest', 'K-Means']
    all_scores = [dt_acc, rf_acc, kmeans_purity]
    all_colors = ['#e74c3c', '#3498db', '#f39c12']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(all_models, all_scores, color=all_colors, edgecolor='black', linewidth=0.5)
    # Annotate each bar with its value
    for bar, score in zip(bars, all_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.title('All Methods Comparison (Accuracy / Purity)')
    plt.ylabel('Score')
    plt.ylim(0, 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/all_methods_comparison.png')
    plt.close()

    # 7b. Confusion Matrices — one for each supervised model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cms = [dt_cm, rf_cm]
    for i, (cm, name) in enumerate(zip(cms, models)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/confusion_matrices.png')
    plt.close()

    # 7c. Feature Importance — shows which features matter most for each tree model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    # Random Forest feature importance (top 20)
    rf_importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=True).tail(20)
    rf_importances.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Random Forest Top 20 Feature Importance')

    # Decision Tree feature importance (top 20)
    dt_importances = pd.Series(dt.feature_importances_, index=selected_features).sort_values(ascending=True).tail(20)
    dt_importances.plot(kind='barh', ax=ax2, color='salmon')
    ax2.set_title('Decision Tree Top 20 Feature Importance')

    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

    # 7d. Multi-metric Comparison — Precision, Recall, F1 side by side
    metrics_df = pd.DataFrame({
        'Model': models * 3,
        'Score': [dt_prec, rf_prec, dt_rec, rf_rec, dt_f1, rf_f1],
        'Metric': ['Precision'] * 2 + ['Recall'] * 2 + ['F1-Score'] * 2
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='Set2')
    plt.title('Supervised Model Performance: Precision, Recall, F1-Score')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('static/multi_metric_comparison.png')
    plt.close()

    # 7e. ROC Curves — one curve per supervised model + random guess baseline
    plt.figure(figsize=(8, 6))

    # Decision Tree ROC curve
    dt_probs = dt.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    auc_dt = auc(fpr_dt, tpr_dt)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='salmon')

    # Random Forest ROC curve
    rf_probs = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='skyblue')

    # Random guess baseline (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curves.png')
    plt.close()

    # ============================================================
    # 8. GRAPHS — K-Means Clustering Visualizations
    # ============================================================
    print(">> [8/8] Generating K-Means clustering graphs...")

    # 8a. K-Means Cluster Distribution — bar chart of samples per cluster
    plt.figure(figsize=(6, 4))
    cluster_ids = list(kmeans_cluster_counts.keys())
    cluster_sizes = list(kmeans_cluster_counts.values())
    bars = plt.bar(
        [f'Cluster {c}' for c in cluster_ids],
        cluster_sizes,
        color=['#3498db', '#e74c3c']
    )
    # Annotate each bar with the count
    for bar, size in zip(bars, cluster_sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 str(size), ha='center', va='bottom', fontweight='bold')
    plt.title('K-Means Cluster Distribution')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('static/kmeans_cluster_dist.png')
    plt.close()

    # 8b. K-Means Clusters vs Actual Labels — heatmap cross-tabulation
    plt.figure(figsize=(6, 4))
    sns.heatmap(kmeans_crosstab, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=le_target.classes_, yticklabels=[f'Cluster {c}' for c in kmeans_crosstab.index])
    plt.title('K-Means Clusters vs Actual Disease Status')
    plt.xlabel('Actual Label')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('static/kmeans_vs_actual.png')
    plt.close()

    # 8c. Elbow Method — find optimal K by plotting inertia for K=2..10
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        temp_km = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_km.fit(X_scaled)
        inertias.append(temp_km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker='o', color='teal')
    plt.title('K-Means Elbow Method (Optimal K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/kmeans_elbow.png')
    plt.close()

    print(">> All models trained and graphs generated successfully!")

    # ============================================================
    # PREPARE DROPDOWN VALUES — for the prediction form
    # ============================================================
    dropdown_vals = {col: encoders[col].classes_.tolist() for col in categorical_cols}

    # ============================================================
    # RETURN RESULTS — bundle everything for the Flask routes
    # ============================================================
    return {
        # Data preview
        'df_sample': df.head(10).to_html(classes='table table-striped', index=False),

        # Supervised model metrics (Decision Tree & Random Forest only)
        'accuracies': {'Decision Tree': dt_acc, 'Random Forest': rf_acc},
        'precisions': {'Decision Tree': dt_prec, 'Random Forest': rf_prec},
        'recalls': {'Decision Tree': dt_rec, 'Random Forest': rf_rec},
        'f1_scores': {'Decision Tree': dt_f1, 'Random Forest': rf_f1},
        'aucs': {'Decision Tree': auc_dt, 'Random Forest': auc_rf},
        'cms': {'Decision Tree': dt_cm.tolist(), 'Random Forest': rf_cm.tolist()},

        # K-Means clustering results (Unsupervised)
        'kmeans': {
            'silhouette_score': kmeans_silhouette,
            'inertia': kmeans_inertia,
            'cluster_counts': kmeans_cluster_counts,
            'crosstab': kmeans_crosstab.to_html(classes='table table-striped'),
            'purity': kmeans_purity,
        },

        # Trained model objects (for prediction)
        'rf_model': rf,
        'dt_model': dt,

        # Encoders and feature info (for prediction form)
        'encoders': encoders,
        'selected_features': selected_features,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'dropdown_vals': dropdown_vals,

        # Raw data (for random data generation on prediction page)
        'raw_df': df_full[selected_features]
    }


# ============================================================
# RUN PROCESSING — train models and generate graphs at startup
# ============================================================
results = process_and_test_models()


# ============================================================
# FLASK ROUTES — serve the web pages
# ============================================================

@app.route('/')
def index():
    """Home page — dataset overview and methodology explanation."""
    return render_template('index.html', data=results)

@app.route('/comparison')
def comparison():
    """Comparison page — model metrics, graphs, and K-Means analysis."""
    return render_template('comparison.html', data=results)

@app.route('/prediction')
def prediction():
    """Prediction page — manual input form and random data feature."""
    return render_template('prediction.html', data=results)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction form submission — uses Random Forest model."""
    try:
        raw_inputs = []
        for col in results['selected_features']:
            val = request.form.get(col)
            if col in results['categorical_cols']:
                le = results['encoders'][col]
                try:
                    encoded_val = le.transform([val])[0]
                except:
                    encoded_val = 0
                raw_inputs.append(encoded_val)
            else:
                raw_inputs.append(float(val))

        input_df = pd.DataFrame([raw_inputs], columns=results['selected_features'])
        # Ensure all columns are numeric for the model
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Use Random Forest for prediction (best-performing supervised model)
        rf = results['rf_model']
        prediction = rf.predict(input_df)[0]

        le_target = results['encoders']['Disease_Status']
        result = le_target.inverse_transform([prediction])[0]

        return render_template('prediction.html', data=results, prediction_text=f'Result: {result}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('prediction.html', data=results, prediction_text=f'Error: {str(e)}')

@app.route('/get_random_data')
def get_random_data():
    """Return a random row from the dataset as JSON (for 'Try Random Data' button)."""
    random_row = results['raw_df'].sample(1).iloc[0].to_dict()
    # Convert all values to string to avoid JSON serializable issues with numpy types
    for key in random_row:
        random_row[key] = str(random_row[key])
    return jsonify(random_row)


# ============================================================
# APP ENTRY POINT — start the Flask development server
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
