import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import hog

# Specify your dataset directory
dataset_dir = 'C:/Users/91866/OneDrive - Amrita Vishwa Vidyapeetham/Desktop/git/live_facial_recognition/face_recognization/dataset'

# Function to load images and labels from the dataset directory
# Function to load images and labels from the dataset directory
def load_dataset(dataset_dir):
    image_data = []
    labels = []
    
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = io.imread(image_path)
            # You can add preprocessing here (e.g., resize, HOG feature extraction)
            image = transform.resize(image, (64, 64))  # Resize to a common size
            image = hog(image)  # Apply HOG feature extraction
            image_data.append(image)
            labels.append(label)
    
    return np.array(image_data), np.array(labels)


# Load the dataset
X, y = load_dataset(dataset_dir)

# Preprocess the data (flatten and normalize)
X = StandardScaler().fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
