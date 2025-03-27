import os
import numpy as np
import pandas as pd
import spectral
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Failed to create RectangleSelector object. Interactive pixel class labeling will be unavailable.")

# Defin the base path and folders
base_path = "/Downloads/"
folders = ['Ra02', 'Ra03', 'Rb02', 'Rb03']
target_shape = (100, 100)  
components_to_keep = 10  
data_summary = []

# Visualization save path
visualization_path = os.path.join(base_path, "visualizations")
os.makedirs(visualization_path, exist_ok=True)

# Function to process each image
def process_image(folder, img_file):
    img_path = os.path.join(base_path, folder, img_file)
    try:
        img = spectral.open_image(img)
        img_data = img.load()


        resized_img = resize(img_data, target_shape, anti_aliasing=True)

        # Flatten image for PCA processing
        flattened_img = resized_img.reshape(-1, resized_img.shape[-1])

        # Pre-PCA scaling using StandardScaler
        scaler = StandardScaler()
        scaled_flattened_img = scaler.fit_transform(flattened_img)

        # Apply PCA
        pca = PCA(n_components=components_to_keep)
        pca_data = pca.fit_transform(scaled_flattened_img)
        pca_img = pca_data.reshape((target_shape[1], target_shape[0], components_to_keep))

        # Post-PCA normalization of the first principal component
        first_pc = pca_img[:, :, 2]
        minmax_scaler = MinMaxScaler(feature_range=(0, 2))
        normalized_first_pc = minmax_scaler.fit_transform(first_pc)

        # Saving a visual representation of the first PCA component
        plt.figure(figsize=(6, 6))
        plt.imshow(normalized_first_pc, cmap='rainbow', norm=Normalize(vmin=0, vmax=1))
        plt.colorbar()
        plt.title(f"First PCA Component (Normalized): {img_file}")
        plt.savefig(os.path.join(visualization_path, f"{folder}_{img_file.split('.')[0]}_pca_visualization.png"))
        plt.close()

       
        mean = np.mean(normalized_first_pc)
        std_dev = np.std(normalized_first_pc)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # Append metadata for the current image
        data_summary.append({
            'Folder': folder,
            'Image': img_file,
            'PCA Transformed Image Shape with 10 components': pca_img.shape,
            'Explained Variance Ratio by Each Component': pca.explained_variance_ratio_.tolist(),
            'Eigenvectors 10 components': pca.components_.tolist(),
            'Cumulative Explained Variance': cumulative_variance.tolist(),
            'Number of components needed to explain 95% of variance': n_components_95,
            'Mean of First Principal Component (Normalized)': mean,
            'Standard Deviation of First Principal Component (Normalized)': std_dev
        })

    except Exception as e:
        print(f"Error opening or processing image {img_path}: {e}")


total_start_time = time.time()

# Processing each folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.hdr')]

    
    start_time = time.time()

    # Process only the first 20 images per folder
    for img_file in image_files[:20]:
        process_image(folder, img_file)

    # End time for processing the folder
    end_time = time.time()

    # Calculate the elapsed time for the folder
    elapsed_time = end_time - start_time
    print(f"Time taken to process 20 images in folder {folder}: {elapsed_time:.2f} seconds")

# End time for the entire process
total_end_time = time.time()


total_elapsed_time = total_end_time - total_start_time
print(f"Total time taken to process all images: {total_elapsed_time:.2f} seconds")


summary_df = pd.DataFrame(data_summary)

# Add notes column for analysis
summary_df['Notes'] = summary_df.apply(
    lambda row: (
        "Better separation in class Ra02" if row['Folder'] == 'Ra02' and row['Mean of First Principal Component (Normalized)'] > 0.5 
        else "Possible overlap with Rb03" if row['Folder'] == 'Ra03' 
        else "Feature distribution needs review"
    ), axis=1
)

# Save metadata to Excel
output_file = os.path.join(base_path, 'pca_metadata_with_analysis.xlsx')
summary_df.to_excel(output_file, index=False)

print(f"Metadata saved to {output_file}")
