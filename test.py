import numpy as np

a = np.identity(4)
print(a.shape)

b = np.zeros((5,2,2))
print(b.shape)

c = np.mreshape(b, a)
