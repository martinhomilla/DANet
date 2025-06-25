from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame
data.to_csv('california_housing.csv', index=False)
import pandas as pd
