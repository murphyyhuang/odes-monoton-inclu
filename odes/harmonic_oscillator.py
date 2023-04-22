# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


class HarmonicOscillator:
  
  def __init__(self):
    
    self.state_dim = 2
    
    self.lower_bd = -2
    self.upper_bd = 2
    
    self.linear_mat = np.array([[0, -1], [1, 0]])
    
    self.random_state = np.random.RandomState(seed=123)
    
  def compute_ne(self):
    return np.zeros((self.state_dim, 1))
  
  def opt(self, state_vec):
    
    return self.linear_mat @ state_vec
  
  def project(self, state_vec):
    
    res_vec = np.maximum(state_vec, self.lower_bd)
    res_vec = np.minimum(res_vec, self.upper_bd)
    
    return res_vec
  
