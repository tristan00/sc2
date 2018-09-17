import numpy as np


p = np.array([1.0, 4.0, 2.0, 2.0])
p /= p.sum()
res = {0:0,
       1:0,
       2:0,
       3:0}
for i in range(1000):
    r = np.random.choice([0, 1, 2, 3], p = p)
    print(r)
    res[int(r)] += 1

print(res)