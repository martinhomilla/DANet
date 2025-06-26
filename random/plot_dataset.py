from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
import umap

def quantile_transform(X):
    quantile_train = np.copy(X)
    qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
    X = qt.transform(X)

data = fetch_california_housing(as_frame=True).frame
data.dropna()
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']
X = StandardScaler().fit_transform(X)
print(X)
reducer = umap.UMAP(n_components=2,random_state=55688, min_dist=0.5)
embedding = reducer.fit_transform(X)

# Crear un DataFrame con el embedding y la variable objetivo para facilitar la visualización
df_umap = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
df_umap['target'] = y

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    x='UMAP_1',
    y='UMAP_2',
    hue='target',       # Colorea los puntos según el valor de la variable objetivo
    palette='viridis',  # Elige una paleta de colores (ej. 'viridis', 'plasma', 'cividis')
    s=20,               # Tamaño de los puntos
    alpha=0.7,          # Transparencia de los puntos
    data=df_umap
)

plt.title('Distribución de tu Dataset de Regresión con UMAP', fontsize=16)
plt.xlabel('Componente UMAP 1', fontsize=12)
plt.ylabel('Componente UMAP 2', fontsize=12)
plt.colorbar(scatter.collections[0], label='Valor de la Variable Objetivo')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

