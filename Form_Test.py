import os
import spectral
import numpy as np
import cv2
from sklearn.decomposition import PCA
import pandas as pd

# Defining the classes and base path for folders
classes = ['Ra02', 'Ra03', 'Rb02', 'Rb03']
base_path = 'C:/Downloads'  # Update path as needed


metrics_data = []

# Loop through each class
for class_name in classes:
    print(f"\nProcessing class: {class_name}")
    class_path = os.path.join(base_path, class_name)
    
    
   
    image_files = [f for f in os.listdir(class_path) if f.endswith('.hdr')]
    if len(image_files) == 0:
        print(f"No .hdr files found in {class_path}.")
        continue

    # Process each image file
    for img_file in image_files[:10]:  # Limit to 10 images per class
        img_path = os.path.join(class_path, img_file)
        print(f"Processing image: {img_file}")

      
        img = spectral.open_image(img_path)
        img_data = img.load()


        pixels = img_data.shape[0] * img_data.shape[1]
        bands = img_data.shape[2]
        reshaped_img = img_data.reshape(pixels, bands)

      
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(reshaped_img)

      
        pc1_img = pca_result.reshape(img_data.shape[0], img_data.shape[1])

        
        pc1_img_uint8 = cv2.normalize(pc1_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply Harris corner detection
        block_size = 2  
        ksize = 3  
        k = 0.06  
        dest = cv2.cornerHarris(pc1_img_uint8, block_size, ksize, k)

        # Dilate to mark the corners
        dest_dilated = cv2.dilate(dest, None)

       
        dest_dilated = cv2.normalize(dest_dilated, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculating metrics for image
        corner_response_min = dest.min()
        corner_response_max = dest.max()
        normalized_corner_response_min = dest_dilated.min()
        normalized_corner_response_max = dest_dilated.max()

        # thresholding
        threshold = 0.01  
        corner_count = np.sum(dest_dilated > threshold * dest_dilated.max())

        # Save the metrics in the list
        metrics_data.append({
            'Class': class_name,
            'Image': img_file,
            'Corner Response Min': corner_response_min,
            'Corner Response Max': corner_response_max,
            'Normalized Corner Response Min': normalized_corner_response_min,
            'Normalized Corner Response Max': normalized_corner_response_max,
            'Number of Detected Corners': corner_count
        })

#saving it to an Excel file
metrics_df = pd.DataFrame(metrics_data)
output_excel_path = 'corner_detection_metrics.xlsx'
metrics_df.to_excel(output_excel_path, index=False)
print(f"Saved all metrics to: {output_excel_path}")
