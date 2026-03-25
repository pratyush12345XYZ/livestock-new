import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Configuration
n_samples = 500

# Generating numerical features
data = {
    'Temperature': np.random.uniform(37.0, 41.0, n_samples),
    'Heart_Rate': np.random.randint(60, 100, n_samples),
    'Respiratory_Rate': np.random.randint(15, 35, n_samples),
    'Age': np.random.randint(1, 15, n_samples),
    'Weight': np.random.uniform(100.0, 600.0, n_samples),
}

# Generating categorical features
data['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
data['Species'] = np.random.choice(['Cow', 'Sheep', 'Goat'], n_samples)
data['Feed_Type'] = np.random.choice(['Grass', 'Grain', 'Silage'], n_samples)

# Creating target variable based on some rules (logic for "diseased")
# Higher temp + higher heart rate + lower weight -> more likely diseased
prob_diseased = (
    (data['Temperature'] - 37) / 4 * 0.4 +
    (data['Heart_Rate'] - 60) / 40 * 0.3 +
    (1 - (data['Weight'] - 100) / 500) * 0.3
)
data['Disease_Status'] = ['Diseased' if p > 0.5 else 'Healthy' for p in prob_diseased]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('livestock_data.csv', index=False)
print("livestock_data.csv generated successfully.")
