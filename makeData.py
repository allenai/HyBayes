import numpy as np
import sys, os
folderName = "fakeExperimentResults"

def getPositiveInput():
  dd = 0.8
  y = [2 * np.random.normal(0, 1, size=40) + 100 - dd,
       2 * np.random.normal(0, 1, size=30) + 100 + dd]
  return y


def getBinaryInput(n1=10, a1=4, n2=15, a2=8):
  x1 = np.zeros(shape=n1, dtype=np.int)
  x1[:a1] = 1

  x2 = np.zeros(shape=n2, dtype=np.int)
  x2[:a2] = 1

  return [x1, x2]


def saveInput(inpFunc, fileNamePrefix):
  y = inpFunc()
  form = "%d" if y[0].dtype == np.int else "%f"
  for i in range(2):
    np.savetxt(f"./{folderName}/{fileNamePrefix}_{i}.csv", y[i], fmt=form)


if __name__ == '__main__':
  if not os._exists(folderName):
    os.mkdir(folderName)
  saveInput(getBinaryInput, fileNamePrefix="BinInp")
  saveInput(getPositiveInput, fileNamePrefix="PositiveInp")
