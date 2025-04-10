Hyperspectral Imaging for Construction Waste Classification
📌 Project Overview
This project focuses on leveraging hyperspectral imaging (HSI) to classify and analyze construction waste components efficiently. Traditional RGB cameras struggle with material differentiation due to limited spectral resolution. Our solution extracts spectral, spatial, and texture features from hyperspectral data to enhance classification accuracy, enabling automated waste sorting for recycling and environmental sustainability.
 Key Objectives
Feature Extraction: Extract spectral, form, and texture features from hyperspectral images.

Dimensionality Reduction: Apply Principal Component Analysis (PCA) to mitigate the Hughes phenomenon and reduce redundancy.

Feature Selection: Use Mutual Information and Support Vector Machines (SVM) to identify the most discriminative features.

Classification: Achieve high-accuracy material classification for construction waste (e.g., bitumen felt, bricks, roofing tiles).

Outlier Analysis: Implement statistical methods (IQR, Z-score) to detect and handle outliers.

🛠️ Tools & Technologies
Python Libraries:

Spectral (HSI processing), scikit-learn (PCA, SVM, Mutual Information), OpenCV (image processing), Pandas (data analysis), Matplotlib/Seaborn (visualization).

Methodologies:

Median Filtering: Noise reduction.

Spectral Unmixing: Endmember analysis.

GLCM: Texture feature extraction.

PCA: Dimensionality reduction.

SVM: Supervised classification (88% accuracy).

📂 Dataset
Source: 60,000 unlabeled hyperspectral images (subset of 80 images used).

Classes:

Ra02, Ra03 (porous/dense bricks), Rb02, Rb03 (bitumen felt, roofing tiles).

Preprocessing: Noise reduction, normalization, and outlier removal.

🔍 Key Insights & Results
Feature Importance:

High mutual information: Standard Deviation, Variance, Peak Reflectance, GLCM Energy.

Low relevance: Aspect Ratio, Kurtosis (discarded).

Classification Performance:

88% overall accuracy using SVM.

Perfect precision/recall for Ra02 and Ra03; challenges with Rb02 due to spectral overlap.

Outlier Handling:

Identified and removed extreme values (e.g., aspect ratio outliers) using IQR/Z-score.

📊 Visualizations (See Full Report)
Feature Distributions: Histograms, box plots.

Spectral Unmixing: 3D endmember plots.

Confusion Matrix: Class-wise performance.

Correlation Matrix: Feature interdependencies.

🚀 Future Work
Segmentation: Improve object-area calculations.

Deep Learning: Explore CNNs for enhanced classification.

Hyperparameter Tuning: Optimize SVM/RF models.

Cross-Validation: Ensure robustness.


