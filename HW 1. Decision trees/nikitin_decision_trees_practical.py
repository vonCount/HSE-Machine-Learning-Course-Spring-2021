# -*- coding: utf-8 -*-
"""

1. Use the Tree data structure below; write code to build the tree from figure 1.2 in Daumé.
"""

class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1

#Creating leaves

leaf1 = Tree.leaf('like')
leaf2 = Tree.leaf('like')
leaf3 = Tree.leaf('like')
leaf4 = Tree.leaf('nah')
leaf5 = Tree.leaf('nah')


branch4 = Tree(data="morning", left=leaf3, right=leaf5)
branch3 = Tree(data='likedOtherSys', left=leaf4, right=leaf2)
branch2 = Tree(data='TakenOtherSys', left=branch4, right=branch3)
tree = Tree(data='isSystems', left=leaf1, right=branch2)

print(tree)

"""2. In your python code, load the following dataset and add a boolean "ok" column, where "True" means the rating is non-negative and "False" means the rating is negative.


* **Attention: we were told during the class that non-negative here stand for 'more then zero', that's why I didn't include zero into the formula.**



"""

#Loading data

import pandas as pd

url = "https://raw.githubusercontent.com/vonCount/HSE-Machine-Learning-Course-Spring-2021/main/data.csv"
data = pd.read_csv(url)

#Adding a column

import numpy as np

#data['ok'] = np.where(data['rating']>0, True, False)

data['ok'] = data.rating >= 0

data

"""3. Write a function which takes a feature and computes the performance of the corresponding single-feature classifier:"""

features = ['easy', 'ai', 'systems', 'theory', 'morning']

def single_feature_score(data, goal, feature):
    yes = data[data[feature] == True][goal]
    no = data[data[feature] == False][goal]
    score = (np.sum(np.bincount(yes).argmax() == yes) + np.sum(np.bincount(no).argmax()== no))/len(data)
    return score

"""Use this to find the best feature.  

*Which* feature is best? Which feature is worst?
"""

def best_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
    return max(features, key=lambda f: single_feature_score(data, goal, f))

def worst_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
    return min(features, key=lambda f: single_feature_score(data, goal, f))

"""4. Implement the DecisionTreeTrain and DecisionTreeTest algorithms from Daumé, returning Trees. (Note: our dataset and his are different; we won't get the same tree.)"""

def DecisionTreeTrain(data, goal, features):
    guess = data[goal].value_counts().idxmax()
    if data[goal].nunique() == 1:
        return Tree.leaf(guess)
    elif len(features) == 0:
        return Tree.leaf(guess)
    else:
        f = best_feature(data, goal, features)
        no = data[data[f] == 'False']
        yes = data[data[f] == 'True']
        features.remove(f)
        left = DecisionTreeTrain(data = no, goal = goal, features = features)
        right = DecisionTreeTrain(data = yes, goal = goal, features = features)   
        return Tree(data = f, left=left, right=right)

def DecisionTreeTest(tree):
    if tree.is_leaf():
        return tree.data
    else: 
        if test_point[tree.data] == False:
            return DecisionTreeTest(tree.left)
        else:
            return DecisionTreeTest(tree.right)

"""I didn't manage to make this code work so I couldn't start the task 5. I hope to get at least some points."""
