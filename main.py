import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/HAM10000_metadata.csv")

# path" /home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2






