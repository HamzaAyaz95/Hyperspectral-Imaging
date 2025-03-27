import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/MI.xlsx"
data = pd.read_excel(file_path)

# Ensure only numerical columns are selected for scaling
numerical_columns = data.select_dtypes(include=[np.number]).columns

# Scaling the numerical feature columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Printing the feature matrix
print("Feature Matrix (first 5 rows):")
print(data.head())


mi_matrix = pd.DataFrame(index=numerical_columns, columns=numerical_columns)

# Calculating the mutual information matrix
for feature1 in numerical_columns:
    for feature2 in numerical_columns:
        if feature1 == feature2:
            mi_matrix.loc[feature1, feature2] = np.nan  
        else:
            mi_score = mutual_info_regression(data[[feature1]], data[feature2])[0]
            mi_matrix.loc[feature1, feature2] = mi_score


mi_matrix = mi_matrix.apply(pd.to_numeric)


print("Mutual Information Matrix:")
print(mi_matrix)


if 'Mean' in mi_matrix.index and 'Percentage Total Object Area' in mi_matrix.columns:
    mi_mean_total_area = mi_matrix.loc['Mean', 'Percentage Total Object Area']
    print(f"Mutual Information between 'Mean' and 'Percentage Total Object Area': {mi_mean_total_area}")
else:
    print("Either 'Mean' or 'Percentage Total Object Area' is not present in the mutual information matrix.")

# Printing shape of MI matrix
print("Shape of the mutual information matrix:", mi_matrix.shape)

# Visualization
plt.figure(figsize=(12, 10))
sns.heatmap(mi_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Mutual Information Matrix Heatmap', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Saving the mutual information matrix to a CSV file
mi_matrix.to_csv("mutual_information_matrix.csv", index=True)
