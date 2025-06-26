 import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

with rasterio.open('GRAY_HR_SR_OB.tif') as src:
    image_arr = src.read()

sub_image = image_arr[0][3120:3170, 13750:13800]
X = sub_image.reshape(-1, 1)

best_score = -1
best_k = 2
best_labels = None

for k in range(2, 7):
    gmm = GaussianMixture(n_components=k, random_state=0)
    labels = gmm.fit_predict(X)
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f'Best n_components: {best_k}, Silhouette Score: {best_score:.3f}')

clustered_img = best_labels.reshape(sub_image.shape)
plt.figure(figsize=(5,5))
plt.title(f'GMM Clustering (n_components={best_k})')
plt.imshow(clustered_img, cmap='tab20')
plt.axis('off')
plt.show()