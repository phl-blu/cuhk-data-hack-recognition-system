import cv2
import numpy as np
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

image_path = "dataset_complete/glass/0ee7f289-a78e-4448-b182-bf5bdc4b9298.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

features = extractor.extract_features(image)

print("single feature vector length:", len(features))
print("Example values:", features[:20])
