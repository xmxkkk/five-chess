from sklearn.preprocessing import Normalizer
import numpy as np

data=Normalizer().fit_transform(np.array([[0.001,1000]]))
print(data)