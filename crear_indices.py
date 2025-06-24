import numpy as np
import pandas as pd
import os
seed = 42

total_samples = 581012
indices = np.arange(total_samples)
np.random.shuffle(indices)

train_size = int(total_samples * 0.7)
test_size = int(total_samples * 0.15)

train_idx = indices[:train_size]
test_idx = indices[train_size:train_size + test_size]
valid_idx = indices[train_size + test_size:]




# Imprime para copiar o guarda en un archivo
pd.DataFrame(train_idx).to_csv('data/forest_cover_type/train_idx.csv', header=False, index=False)
pd.DataFrame(test_idx).to_csv('data/forest_cover_type/test_idx.csv', header=False, index=False)
pd.DataFrame(valid_idx).to_csv('data/forest_cover_type/valid_idx.csv', header=False, index=False)

