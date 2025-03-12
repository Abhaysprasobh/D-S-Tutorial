import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define file paths (Ensure all files are in the same directory)
dataset_path = "employee_promotion_dataset.csv"
vp_data_path = "unknown_vps.csv"

# Step 1: Generate and Save Dataset (if not already created)
np.random.seed(42)
num_samples = 500

experience = np.random.randint(1, 21, num_samples)  # Years of experience (1-20)
education = np.random.choice([1, 2, 3, 4], num_samples)  # Education levels (1-4)
performance_score = np.random.randint(1, 11, num_samples)  # Performance rating (1-10)
past_promotions = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 70% No, 30% Yes
training_hours = np.random.randint(10, 100, num_samples)  # Training hours (10-100)

# Compute promotion probability based on weighted factors
promotion_prob = (
    0.3 * (experience / 20) +
    0.3 * (education / 4) +
    0.2 * (performance_score / 10) +
    0.1 * past_promotions +
    0.1 * (training_hours / 100)
)

# Convert probability to binary labels (1: Promoted, 0: Not Promoted)
promoted = (promotion_prob > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Experience': experience,
    'Education': education,
    'Performance_Score': performance_score,
    'Past_Promotions': past_promotions,
    'Training_Hours': training_hours,
    'Promoted': promoted
})

# Save dataset to CSV
df.to_csv(dataset_path, index=False)

# Step 2: Load Dataset for Training
df = pd.read_csv(dataset_path)

# Splitting dataset into features (X) and target (y)
X = df.drop(columns=['Promoted'])
y = df['Promoted']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate Model Performance
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Create Unknown VPs Data (if not already provided)
unknown_vps = pd.DataFrame({
    'Experience': [15, 18, 10],
    'Education': [4, 3, 4],
    'Performance_Score': [9, 7, 8],
    'Past_Promotions': [1, 0, 1],
    'Training_Hours': [80, 50, 90]
})

# Save unknown VPs data (if needed)
unknown_vps.to_csv(vp_data_path, index=False)

# Step 7: Load Unknown VPs and Predict
unknown_vps = pd.read_csv(vp_data_path)

# Standardize VP data
unknown_vps_scaled = scaler.transform(unknown_vps.drop(columns=['Predicted_Promotion'], errors='ignore'))

# Predict promotions
vp_predictions = model.predict(unknown_vps_scaled)
unknown_vps["Predicted_Promotion"] = vp_predictions

# Save updated predictions
unknown_vps.to_csv(vp_data_path, index=False)

# Display predictions
print("\nUpdated Predictions for Unknown VPs:")
print(unknown_vps)
