import torch
import numpy as np

data = None
for i in range(1, 3):
    f = 'tor/{}.pth'.format(i)
    new = torch.load(f)
    print(new.shape)
    exit(0)
    if data is None:
        data = new
        continue
    else:
        eq = (data == new).all()
        print(eq)
        if np.array_equal(data, new, equal_nan=True):
            print('chist')
        else:
            print('sxal')
