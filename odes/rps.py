# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxpy as cp
import numpy as np


class RPS:
  
  def __init__(self):
    self.state_dim = 6
    
    self.param_d, self.param_l, self.param_w = 0, -1, 1
    mat_A = np.array([
      [self.param_d, self.param_l, self.param_w],
      [self.param_w, self.param_d, self.param_l],
      [self.param_l, self.param_w, self.param_d]
    ])
    self.linear_mat = np.zeros((self.state_dim, self.state_dim))
    self.linear_mat[:self.state_dim//2, :self.state_dim//2] = mat_A.copy()
    self.linear_mat[self.state_dim // 2:self.state_dim, self.state_dim // 2:self.state_dim] \
      = mat_A.T.copy()
    
    self.random_state = np.random.RandomState(seed=123)
  
  def compute_ne(self):
    return np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3]).reshape((-1, 1))
  
  def opt(self, state_vec):
    return self.linear_mat @ state_vec
  
  def project_probability_simplex(self, dec_vec):
    dec_vec_opt = cp.Variable((self.state_dim//2, 1))
    obj = 1/2 * cp.quad_form(dec_vec_opt, np.identity(self.state_dim // 2)) \
      - dec_vec.T @ dec_vec_opt
    constraints = [
      dec_vec_opt >= 0,
      dec_vec_opt <= 1,
      np.ones_like(dec_vec).T @ dec_vec_opt == 1,
    ]
    
    prob = cp.Problem(
      cp.Minimize(obj),
      constraints
    )
    prob.solve(
      solver=cp.OSQP,
      eps_abs=1e-5, eps_rel=1e-5,
      verbose=False
    )
    
    return dec_vec_opt.value
    
  def project(self, state_vec):
    res_vec = np.zeros_like(state_vec)
    res_vec[:self.state_dim // 2, :] = self.project_probability_simplex(
      state_vec[:self.state_dim // 2]
    )
    res_vec[self.state_dim // 2: self.state_dim, :] = self.project_probability_simplex(
      state_vec[self.state_dim // 2: self.state_dim]
    )
    
    return res_vec

