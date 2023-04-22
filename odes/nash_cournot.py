# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cvxpy as cp
import logging
import numpy as np
from scipy.linalg import block_diag


class NashCournot(object):
  
  def __init__(self):
    
    self.player_num = 10
    self.prod_cnt = 10
    self.local_dim = 6
    self.state_dim = self.player_num * self.local_dim
    self.random_state = np.random.RandomState(seed=123)
    self.unit_price_vec = self.random_state.uniform(low=20, high=28, size=(self.state_dim, 1))
    self.mat_lambda = np.diag(self.random_state.uniform(low=0.2, high=0.4, size=(self.prod_cnt, )))
    self.upper_bdd = 10 * np.ones(shape=(self.state_dim, 1))
    
    blkd_mats = []
    global_mat_a = np.empty((self.prod_cnt, 0))
    for i_plr in range(self.player_num):
      
      local_mat_a = self.random_state.uniform(low=0.1, high=0.3, size=(self.prod_cnt, self.local_dim))
      blkd_mats.append(local_mat_a.T @ self.mat_lambda @ local_mat_a)
      global_mat_a = np.concatenate((global_mat_a, local_mat_a), axis=1)
    
    mat1 = block_diag(*blkd_mats)
    mat2 = global_mat_a.T @ self.mat_lambda @ global_mat_a
    
    self.linear_mat = mat1 + mat2
  
  def compute_ne(self):
    dec_vec = cp.Variable((self.state_dim, 1))
    obj = 1/2 * cp.quad_form(dec_vec, self.linear_mat) - self.unit_price_vec.T @ dec_vec
    constraints = [
      dec_vec >= 0,
      dec_vec <= self.upper_bdd,
    ]
    
    prob = cp.Problem(
      cp.Minimize(obj),
      constraints
    )
    prob.solve()
    
    return dec_vec.value
  
  def opt(self, state_vec):
    return self.linear_mat @ state_vec - self.unit_price_vec

  def resolvent(self, cur_vec, lambda_val):
    dec_vec = cp.Variable((self.state_dim, 1))
    obj = 1/2 * cp.quad_form(dec_vec, self.linear_mat) - self.unit_price_vec.T @ dec_vec \
          + cp.quad_form(dec_vec, 1/(2 * lambda_val) * np.identity(self.state_dim)) - 1/lambda_val * cur_vec.T @ dec_vec
    print(1/lambda_val)
    constraints = [
      dec_vec >= 0,
      dec_vec <= self.upper_bdd,
    ]
    prob = cp.Problem(
      cp.Minimize(obj),
      constraints
    )
    try:
      # TODO: try circumventing the following error reported by scipy
      # scipy.sparse.linalg.eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during
      # a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.
      # A personal guess is that is is related with random number for computing eigenvalues; since skipping one iterate
      # and repeating the computation seems to resolve this error.
      prob.solve(
        solver=cp.OSQP,
        eps_abs=1e-5, eps_rel=1e-5,
        verbose=False
      )

      return dec_vec.value
    except:
      logging.warning("ArpackError detected: try skip this iterate")
      return cur_vec

  def project(self, state_vec):
    
    res_vec = np.maximum(state_vec, 0)
    res_vec = np.minimum(res_vec, self.upper_bdd)
    
    return res_vec
  

if __name__ == '__main__':
  
  nash_inst = NashCournot()
  nash_inst.opt(np.zeros((nash_inst.state_dim, 1)))
  