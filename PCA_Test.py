import spectral
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Failed to create RectangleSelector object. Interactive pixel class labeling will be unavailable.")

# Define paths to folders for each class
classes = {
    'Ra02': "C:/Users/anusa/Downloads/",
    'Ra03': "C:/Users/anusa/Downloads/",
    'Rb02': "C:/Users/anusa/Downloads/",
    'Rb03': "C:/Users/anusa/Downloads/"
}

# Initialize an empty DataFrame to store metadata for all classes
meta_data = pd.DataFrame(columns=[
    'Class', 'Image', 'PCA Transformed Image Shape', 'Explained Variance Ratio',
    'Eigenvectors (First 10 components)', 'Cumulative Explained Variance', 
    'Components for 95% Variance', 'Mean First PC', 'Std First PC'
])

# Loop through each class 
for class_name, folder_path in classes.items():
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.hdr')][:20]  # Get the first 20 .hdr files
    
    for image_file in image_files:
        # Load the hyperspectral image
        img = spectral.open_image(os.path.join(folder_path, image_file))
        img_data = img.load()

       
        n_pixels = img_data.shape[0] * img_data.shape[1]
        n_bands = img_data.shape[2]
        reshaped_data = img_data.reshape((n_pixels, n_bands))

        # Applying PCA with 10 components
        pca = PCA(n_components=10)
        pca_data = pca.fit_transform(reshaped_data)

        # Reshape PCA-transformed data back to image dimensions
        pca_img = pca_data.reshape((img_data.shape[0], img_data.shape[1], 10))

        # Metadata extraction
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_95 = np.argmax(cumulative_variance >= 0.90) + 1  # Components for 95% variance
        mean_first_pc = np.mean(pca_img[:, :, 2])
        std_first_pc = np.std(pca_img[:, :, 2])
        eigenvectors = pca.components_  # Eigenvectors (loadings) for PCA

        # Creating a dictionary for the current image metadata
        image_metadata = {
            'Class': class_name,
            'Image': image_file,
            'PCA Transformed Image Shape': pca_img.shape,
            'Explained Variance Ratio': explained_variance_ratio,
            'Eigenvectors (First 10 components)': eigenvectors,
            'Cumulative Explained Variance': cumulative_variance,
            'Components for 95% Variance': n_components_95,
            'Mean First PC': mean_first_pc,
            'Std First PC': std_first_pc
        }

       
        meta_data = pd.concat([meta_data, pd.DataFrame([image_metadata])], ignore_index=True)

# Save the metadata to an Excel file
output_path = '"C:/Users/anusa/Downloads/pca_metadata.xlsx'
meta_data.to_excel(output_path, index=False)

print(f"Metadata has been saved to {output_path}.")
