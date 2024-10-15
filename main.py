import kagglehub

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)
