import numpy as np
import sys, os
folderName = "ArtificialData"

def getPositiveInput(n1=40, n2=30):
  dd = 0.8
  y = [2 * np.random.normal(0, 1, size=n1) + 100 - dd,
       2 * np.random.normal(0, 1, size=n2) + 100 + dd]
  return y


def getBinaryInput(n1=25, a1=12, n2=28, a2=20):
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


def getCountInput(n1=67, lambda1=11.1, n2=76, lambda2=9.9):
 x1 = np.random.poisson(lam=lambda1, size=n1)
 x2 = np.random.poisson(lam=lambda2, size=n2)
 return [x1, x2]


def getOrdinalInput(f1 = (10, 9, 11), f2 = (8, 9, 7)):
 x1 = np.zeros(shape=sum(f1), dtype=np.int)
 for i in range(1, len(f1)):
   begin = sum(f1[:i])
   x1[begin: (begin+f1[i])] = i

 x2 = np.zeros(shape=sum(f2), dtype=np.int)
 for i in range(1, len(f2)):
   begin = sum(f2[:i])
   x2[begin: (begin + f2[i])] = i
 return [x1, x2]


def getBinomialInput(n1=20, p1=0.44, n2=25, p2=0.52):
  x1 = np.zeros(shape=(n1,2), dtype=np.int)
  x1[:, 0] = np.random.poisson(lam=4, size=n1)
  x1[:, 1] = np.random.binomial(n = x1[:, 0], p = p1)

  x2 = np.zeros(shape=(n2, 2), dtype=np.int)
  x2[:, 0] = np.random.poisson(lam=4, size=n2)
  x2[:, 1] = np.random.binomial(n=x2[:, 0], p=p2,)
  return [x1, x2]


if __name__ == '__main__':
  if not os.path.exists(folderName):
    os.mkdir(folderName)
  saveInput(getBinaryInput, fileNamePrefix="BinaryData")
  saveInput(getPositiveInput, fileNamePrefix="PositiveRealData")
  saveInput(getCountInput, fileNamePrefix="CountData")
  saveInput(getOrdinalInput, fileNamePrefix="OrdinalData")
  saveInput(getBinomialInput, fileNamePrefix="BinomialData")

