import spectral
import matplotlib
import os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


# Define the base directory where your images are stored
base_dir = "C:/Downloads/"  
# Define the classes
classes = ['Ra02', 'Ra03', 'Rb02', 'Rb03']
num_images_per_class = 60 # Set the number of images per class

# Function to map data type code to the corresponding format
def get_data_format(data_type):
    data_type_map = {
        1: 'int8',
        2: 'int16',
        3: 'int32',
        4: 'float32',
        5: 'float64',
        12: 'uint16',
        13: 'uint32',
        14: 'complex64',
        15: 'complex128'
    }
    return data_type_map.get(data_type, f'unknown (raw value: {data_type})')

# calculate area
def calculate_area(rows, samples):
    return rows * samples

# Function to apply a median filter to image data
def apply_median_filter(img_data, size=3):
    filtered_data = np.empty_like(img_data)
    for band in range(img_data.shape[2]):
        # Apply median filter to each spectral band
        filtered_data[:, :, band] = median_filter(img_data[:, :, band], size=size)
    return filtered_data

# Function to process each image
def process_image(image_path, apply_filter=True):
    img = spectral.open_image(image_path)
    img_data = img.load()

    if apply_filter:
        # Apply median filter to the image data
        img_data = apply_median_filter(img_data)

    metadata = img.metadata
    rows = img_data.shape[0]
    samples = img_data.shape[1]
    bands = img_data.shape[2]
    interleave = metadata['interleave']
    data_type = int(metadata['data type'])
    quantization_bits = data_type * 8
    data_format = get_data_format(data_type)

    # Calculate basic statistics
    mean = np.mean(img_data)
    std_dev = np.std(img_data)
    variance = np.var(img_data)          # Variance
    area = calculate_area(rows, samples)

    return {
        "Rows": rows,
        "Samples": samples,
        "Bands": bands,
        "Interleave": interleave,
        "Quantization (bits)": quantization_bits,
        "Data format": data_format,
        "Mean": mean,
        "Standard deviation": std_dev,
        "Variance": variance,  # Add variance to the metadata
        "Area": area
    }


def gather_class_metadata(class_name):
    class_dir = os.path.join(base_dir, class_name)
    metadata_list = []

    # Read all .hdr files in the directory
    for i, filename in enumerate(os.listdir(class_dir)):
        if filename.endswith('.hdr') and i < num_images_per_class:
            image_path = os.path.join(class_dir, filename)
            metadata = process_image(image_path, apply_filter=True)  # Apply filter before extracting metadata
            metadata['Class'] = class_name
            metadata['Image'] = filename
            metadata_list.append(metadata)

    return metadata_list


all_metadata = []
for class_name in classes:
    class_metadata = gather_class_metadata(class_name)
    all_metadata.extend(class_metadata)


df = pd.DataFrame(all_metadata)

# Save metadata to an Excel file
output_excel_file = "image_metadata_variance_median.xlsx"
df.to_excel(output_excel_file, index=False)

print(f"Metadata saved to {output_excel_file}")
