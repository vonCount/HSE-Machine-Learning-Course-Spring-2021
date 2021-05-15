# -*- coding: utf-8 -*-
"""
For this practical, numpy and other numerical libraries are forbidden. You may use only Python standard libraries and code you write and submit yourself.

Tasks:

1. Implement your own Scalar and Vector classes, without using any other modules:
"""

from typing import Union, List
from math import sqrt

class Scalar:
  def __init__(self: Scalar, val: float):
    self.val = float(val)
  def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
    if isinstance(other, Vector):
      res = []
      for i in range(len(other.entries)):
        res.append(self.val * other.entries[i])
      return Vector(*res)
    elif isinstance(other, Scalar):
        return Scalar(self.val * other.val)
    else:
      print("It's neiter a Vector nor a Scalar")
  def __add__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val + other.val)
  def __sub__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val - other.val)
  def __truediv__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val / other.val)
  def __rtruediv__(self: Scalar, other: Vector) -> Vector:
    res = []
    for i in range(len(other)):
      x = other.entries[i]/self.val
      res.append(x)
    return Vector(*res)
  def __repr__(self: Scalar) -> str:
    return "Scalar(%r)" % self.val
  def sign(self: Scalar) -> int:
    if self.val == 0:
      return 0
    elif self.val < 0:
      return -1
    else:
      return 1
  def __float__(self: Scalar) -> float:
    return self.val

class Vector:
  def __init__(self: Vector, *entries: List[float]):
    self.entries = entries
  def zero(size: int) -> Vector:
    return Vector(*[0 for i in range(size)])
  def __add__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] + other.entries[i])
    return Vector(*res)
  def __sub__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] - other.entries[i])
    return Vector(*res)
  def __mul__(self: Vector, other: Vector) -> Scalar:
    res_v = 0
    if len(self.entries) == len(other.entries):
      for i in range(len(self.entries)):
        res_v += self.entries[i] * other.entries[i]
    return Scalar(res_v)
  def magnitude(self: Vector) -> Scalar:
    res = 0
    for i in range(len(self.entries)):
      res += self.entries[i]**2
    return Scalar(sqrt(res))
  def unit(self: Vector) -> Vector:
    return self / self.magnitude()
  def __len__(self: Vector) -> int:
    return len(self.entries)
  def __repr__(self: Vector) -> str:
    return "Vector%s" % repr(self.entries)
  def __iter__(self: Vector):
    return iter(self.entries)

"""2. Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. Do not permute the dataset when training; run through it linearly.
(Hint on how to use the classes: make w and x instances of Vector, y and b instances of Scalar. What should the type of D be? Where do you see the vector operation formulas?)


"""

def PerceptronTrain(D, Maxiter = 100):
  """ returns the weights for the perceptron """
  # initiate essentials
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(Maxiter):
    # process the first vector
    for x, y in D:
      # compute activation
      a = x * w + b
      # check if label and activation have different signs
      if (y * a).sign() <= 0: 
        w += y * x
        b += y
  return w, b

def PerceptronTest(w, b, D):
    """ 
    takes weights, bias and vectors 
    and returns their labels
    """
  res = []
  for x,y in D:
    # compute activate
    a = x * w + b
    res.append(a.sign())
  return res

"""3. Make a 90-10 test-train split and evaluate your algorithm on the following dataset:"""

from random import randint
from random import shuffle

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

def merge(l1, l2):       
    merged_list = [(l1[i], l2[i]) for i in range(0, len(l1))] 
    return merged_list      
    
def train_test_split(data, split, shuf = True):
  if shuf = True:
    shuffle(data)
  sep = int(len(data)*split)
  train = data[:sep]
  test= data[sep:]
  return train, test

D = merge(xs, ys)

train = train_test_split(D, 0.9, shuf = False)[0]
test = train_test_split(D, 0.9, shuf = False)[1]

w, b = PerceptronTrain(train)
y_pred = PerceptronTest(w, b, test)

def score(y_pred, y_true):
  size = len(y_true)
  true = 0
  for i in range(all):
    if y_pred[i] == y_true[i][1].sign():
      true += 1
  return true/size*100

print(score(y_pred, test))

"""You should get that w is some multiple of v, and the performance should be very good. (Some noise is introduced by the last factor in y.)

4. Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:
"""

from random import randint
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else -1 for x in xs]

xs_xor = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys_xor = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs_xor]

D_xor = merge(xs_xor, ys_xor)

train_xor, test_xor = train_test_split(D_xor, 0.9, shuf = False)

w, b = PerceptronTrain(train_xor)
y_pr_xor = PerceptronTest(w, b, test_xor)

print(score(y_pr_xor, test_xor))

"""You should get some relatively random w, and the performance should be terrible.

5. Sort the training data from task 3 so that all samples with y < 0 come first, then all samples with y = 0, then all samples with y > 0. (That is, sort by y.)

Graph the performance (computed by PerceptronTest) on both train and test sets versus epochs for perceptrons trained on  

- no permutation  
- random permutation at the beginning  
- random permutation at each epoch  
(This replicates graph 4.4 from Daumé.)
"""

import matplotlib.pyplot as plt
from random import randint, seed, sample
seed(48)

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

train_temp = train
train_sorted = sorted(train, key= lambda x: x[1].val)

perf_no_perm = []
perf_perm_begin = []
perf_perm_each = []


def PerceptronTrainPermutate(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(maxiter):
    shuffle(D)
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
  return w, b


for Maxiter in range(1, 100, 5):
    w, b = PerceptronTrain(train_sorted, Maxiter=i)
    preds = PerceptronTest(w, b, test)
    perf_no_perm.append(score(preds, test))

    shuffle(train_temp)
    w, b = PerceptronTrain(train_temp, Maxiter=i)
    preds = PerceptronTest(w, b, test)
    perf_perm_begin.append(score(preds, test))

    w, b = PerceptronTrainPermutate(train, train_temp, Maxiter=i)
    preds = PerceptronTest(w, b, test)
    perf_perm_each.append(score(preds, test))

x = [x for x in range(1, 100, 5)]
plt.plot(x, perf_no_perm,
         color='blue', label='no permutation', marker='o')
plt.plot(x, perf_perm_begin, color='red',
         label='random permutation at the beginning', marker='o')
plt.plot(x, perf_perm_each,
         color='green', label='random permutation at each epoch', marker='o')

plt.xlabel('Number of epochs to train', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.show()

"""6. Implement AveragedPerceptronTrain; using random permutation at each epoch, compare its performance with PerceptronTrain using the dataset from task 3."""

def AveragedPerceptronTrain(D, Maxiter = 100):
    """ 
    returns the weights for the perceptron 
    """
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)
    u = Vector.zero(len(D[0][0]))
    beta = Scalar(0)
    c = Scalar(1)
    for i in range(Maxiter):
      shuffle(D)
      for x, y in D:
        a = x * w + b
        if (y * a).sign() <= 0: 
          w += y * x
          b += y
          u += y * c * x
          beta += y * c
        c += Scalar(1)
    return w - (Scalar(1)/c) * u, b - beta * (Scalar(1)/c)
  
  """The averaged one works better"""
