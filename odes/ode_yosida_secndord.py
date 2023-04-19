# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from nash_cournot import NashCournot
from scipy.integrate import odeint


def sim_ode_yosida_secndord(prob_inst, t_len, pt_cnt=100, param_alpha=3.0, param_epsilon=3.0):
  
  def ode_yosida_secndord(x_vec, t):

    state_dim = prob_inst.state_dim
    state_vec = x_vec[:state_dim].copy().reshape((-1, 1))
    derv_vec = x_vec[state_dim:].copy().reshape((-1, 1))
    
    # dot state_vec
    vec_field = np.zeros_like(x_vec)
    vec_field[:state_dim] = x_vec[state_dim:].copy()
    
    # dot derv_vec
    lambda_val = ((1 + param_epsilon) / param_alpha**2) * t**2
    derv_vec_updt = -param_alpha/t * derv_vec - 1/lambda_val * (
      state_vec - prob_inst.resolvent(state_vec, lambda_val)
    )
    vec_field[state_dim:] = derv_vec_updt.reshape((-1,))
    
    return vec_field
  
  x_vec_init = prob_inst.random_state.uniform(
    low=0, high=1, size=(prob_inst.state_dim * 2, 1)
  )
  x_vec_init[:prob_inst.state_dim, :] = prob_inst.project(
    x_vec_init[:prob_inst.state_dim])
  x_vec_init = x_vec_init.reshape((-1, ))
  t_range = np.linspace(t_len/pt_cnt + 10, t_len + 10, pt_cnt)
  flow = odeint(ode_yosida_secndord, x_vec_init, t_range)
  
  return flow


def main():
  t_len = 1000
  
  prob_inst = NashCournot()
  sol = prob_inst.compute_ne().reshape((1, -1))
  
  # Trail 1
  data_save_path = './pkl/yosida_{}_{}.pkl'
  param_alpha, param_epsilon = 20, 20
  traject = sim_ode_yosida_secndord(prob_inst, t_len, pt_cnt=5000, param_alpha=param_alpha, param_epsilon=param_epsilon)
  
  diff = traject[:, :prob_inst.state_dim] - sol
  diff_norm = np.linalg.norm(diff, axis=1, ord=2)
  print(diff_norm)
  save_dict = {'diff_norm': diff_norm}
  with open(data_save_path.format(param_alpha, param_epsilon), 'wb') as pkl_writer:
    pickle.dump(save_dict, pkl_writer)

  # Trail 2
  param_alpha, param_epsilon = 10, 10
  traject = sim_ode_yosida_secndord(prob_inst, t_len, pt_cnt=5000, param_alpha=param_alpha, param_epsilon=param_epsilon)

  diff = traject[:, :prob_inst.state_dim] - sol
  diff_norm = np.linalg.norm(diff, axis=1, ord=2)
  print(diff_norm)
  save_dict = {'diff_norm': diff_norm}
  with open(data_save_path.format(param_alpha, param_epsilon), 'wb') as pkl_writer:
    pickle.dump(save_dict, pkl_writer)


if __name__ == '__main__':
  main()