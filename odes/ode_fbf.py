# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from nash_cournot import NashCournot
from harmonic_oscillator import HarmonicOscillator
from rps import RPS
from scipy.integrate import odeint


def sim_ode_fbf(prob_inst, t_len, pt_cnt=5000, param_epsilon=0.1, params_gamma=(0.02, -0.5)):
  def ode_fbf(x_vec, t):
    state_vec = x_vec.reshape((-1, 1))
    grad_info = prob_inst.opt(state_vec)
    gamma_val = param_epsilon + params_gamma[0] * t**params_gamma[1]
    z_vec = prob_inst.project(state_vec - gamma_val * grad_info)
    vec_field = z_vec - state_vec \
      + gamma_val * prob_inst.opt(state_vec) - gamma_val * prob_inst.opt(z_vec)
    
    return vec_field.reshape((-1,))
  
  x_vec_init = prob_inst.random_state.uniform(
    low=0, high=1, size=(prob_inst.state_dim, 1)
  )
  x_vec_init = prob_inst.project(x_vec_init).reshape((-1,))
  t_range = np.linspace(t_len/pt_cnt, t_len, pt_cnt)
  flow = odeint(ode_fbf, x_vec_init, t_range)
  
  return flow


def main():
  t_len = 1000
  pt_cnt = 5000

  prob_inst_dict = {
    'nash_corunot': NashCournot,
    'harmonic_oscillator': HarmonicOscillator,
    'rps': RPS,
  }

  prob_inst_str = "rps"
  prob_inst = prob_inst_dict[prob_inst_str]()
  sol = prob_inst.compute_ne().reshape((1, -1))

  data_save_path = './pkl/fbf_{}_{}_{}_{}.pkl'

  # Trails
  param_list = [(0.1, 0.02, -0.5), (0.05, 0.02, -0.5), (0.1, 0.1, -0.5)]

  for param_epsilon, params_gamma_1, params_gamma_2 in param_list:

    traject = sim_ode_fbf(
      prob_inst, t_len, pt_cnt=pt_cnt,
      param_epsilon=param_epsilon,
      params_gamma=(params_gamma_1, params_gamma_2)
    )
    
    diff = traject - sol
    diff_norm = np.linalg.norm(diff, axis=1, ord=2)
    print(diff_norm)
    save_dict = {'diff_norm': diff_norm, 'traject': traject}
    with open(data_save_path.format(
        prob_inst_str, param_epsilon, params_gamma_1, params_gamma_2
      ), 'wb') as pkl_writer:
      pickle.dump(save_dict, pkl_writer)


if __name__ == '__main__':
  main()
