import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

class FeatureExtractor:
    def __init__(self, hog_pixels_per_cell=(16, 16), hog_cells_per_block=(2, 2),
                 hog_orientations=9, lbp_radius=2, lbp_points=16,
                 hist_bins=(16, 8, 4)):
        # histogram of Oriented Gradients good with edges
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

        # local Binary Patterns good with texture smooth vs rough
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

        # color histogram bins (H, S, V)
        self.hist_bins = hist_bins

    def extract_hog(self, gray):
        features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        return features

    def extract_color_hist(self, img):
        #hsv color space is better for color histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_histogram = cv2.calcHist([hsv], [0, 1, 2],
                            None,
                            self.hist_bins,
                            [0, 180, 0, 256, 0, 256])
        color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()
        return color_histogram

    def extract_lbp(self, gray):
        lbp = local_binary_pattern(gray,P=self.lbp_points,R=self.lbp_radius,method='uniform')
        #_ to skip bin_edges return value
        lbp_histogram, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, self.lbp_points + 3),
                               range=(0, self.lbp_points + 2))
        lbp_histogram = lbp_histogram.astype("float")
        lbp_histogram /= lbp_histogram.sum() + 1e-6
        return lbp_histogram

    def extract_features(self, image):
        # resize image to have const size
        image = cv2.resize(image, (224, 224))
        gray = rgb2gray(image)
        
        hog_features = self.extract_hog(gray)
        color_features = self.extract_color_hist(image)
        lbp_features = self.extract_lbp(gray)

        # Combine all features
        feature_vector = np.concatenate([hog_features, color_features, lbp_features])
        return feature_vector
    