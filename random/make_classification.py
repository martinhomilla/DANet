from sklearn.datasets import make_classification
X, y = make_classification( random_state=42, n_features=3, n_redundant=0 , n_classes=4, n_samples=1000, n_clusters_per_class=1, n_informative=3, class_sep=3)
X.shape
y.shape
list(y[:5])
print(X[:5])
print(y[:5])
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k')
ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.set_zlabel('Característica 3')
ax.set_title('Distribución de Clases')
plt.show()