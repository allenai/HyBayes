import numpy as np


if __name__ == '__main__':
  nHeads = 2
  nTotal = 4
  x = np.zeros(shape=nTotal)
  x[:nHeads]=1
  np.savetxt(f"./data/heads{nHeads}of{nTotal}.csv", x, fmt="%d")