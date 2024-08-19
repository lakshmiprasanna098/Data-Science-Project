import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score

# Sample Data Preparation (Ensure this is a DataFrame)
df = pd.read_csv("Global EV Data 2024.csv") 

 # Convert the dictionary to a DataFrame

# Calculate Growth Rate
df['Growth_Rate'] = df['value'].pct_change() * 100  # Calculate percentage change
df['Growth_Rate'].fillna(0, inplace=True)  # Replace NaN with 0

# Detect Outliers Using Z-Score
df['Z_Score'] = np.abs(stats.zscore(df['Growth_Rate']))  # Calculate Z-score
df['Pred_Outlier'] = df['Z_Score'] > 2  # Identify outliers (Z-score > 2)

# Synthetic True Outliers (for demonstration purposes)
true_outlier_years = [2011, 2015, 2023]  # Assuming these years are known outliers
df['True_Outlier'] = df['year'].isin(true_outlier_years)  # Mark true outliers

# Filter Predicted and True Outliers
Pred_outliers = df[df['Pred_Outlier']]  # DataFrame of predicted outliers
true_outliers = df[df['True_Outlier']]  # DataFrame of true outliers

# Calculate Accuracy Score
accuracy = accuracy_score(df['True_Outlier'], df['Pred_Outlier'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['Growth_Rate'], label='Growth Rate', marker='o')  # Plot Growth Rate
plt.scatter(Pred_outliers['year'], Pred_outliers['Growth_Rate'], color='red', label='Predicted Outliers', marker='x')
plt.scatter(true_outliers['year'], true_outliers['Growth_Rate'],  color='green', label='True Outliers', marker='o')
plt.xlabel('year')
plt.ylabel('Growth Rate (%)')
plt.title('EV Sales Growth Rate (2010-2024) with Outliers')
plt.legend()
plt.grid(True)
plt.show()

# Display Accuracy
print(f"Accuracy Score: {accuracy:.2f}")
