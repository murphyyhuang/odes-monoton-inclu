# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from nash_cournot import NashCournot
from scipy.integrate import odeint


def sim_ode_proj_firstord(prob_inst, t_len, pt_cnt=5000, param_lambda=0.5, param_alpha=0.5):
  
  def ode_proj_firstord(x_vec, t):
    state_vec = x_vec.reshape((-1, 1))
    grad_info = prob_inst.opt(state_vec)
    proj_res = prob_inst.project(state_vec - param_alpha * grad_info)
    vec_field = param_lambda * (proj_res - state_vec)
    
    return vec_field.reshape((-1, ))

  x_vec_init = prob_inst.random_state.uniform(
    low=0, high=1, size=(prob_inst.state_dim, 1)
  )
  x_vec_init = prob_inst.project(x_vec_init).reshape((-1, ))
  t_range = np.linspace(t_len/pt_cnt, t_len, pt_cnt)
  flow = odeint(ode_proj_firstord, x_vec_init, t_range)
  
  return flow


def main():
  t_len = 1000
  pt_cnt = 5000
  
  prob_inst = NashCournot()
  sol = prob_inst.compute_ne().reshape((1, -1))
  
  data_save_path = './pkl/proj_{}_{}.pkl'
  
  # Trails
  param_list = [(0.2, 0.2), (0.5, 0.5), (1, 1)]
  
  for param_lambda, param_alpha in param_list:
    traject = sim_ode_proj_firstord(prob_inst, t_len, pt_cnt=pt_cnt, param_lambda=param_lambda, param_alpha=param_alpha)
    
    diff = traject - sol
    diff_norm = np.linalg.norm(diff, axis=1, ord=2)
    print(diff_norm)
    save_dict = {'diff_norm': diff_norm}
    with open(data_save_path.format(param_lambda, param_alpha), 'wb') as pkl_writer:
      pickle.dump(save_dict, pkl_writer)


if __name__ == '__main__':
  main()