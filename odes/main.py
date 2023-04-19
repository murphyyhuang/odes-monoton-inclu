# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from nash_cournot import NashCournot
from ode_proj_firstord import sim_ode_proj_firstord


def main():
  
  t_len = 1000
  
  prob_inst = NashCournot()
  sol = prob_inst.compute_ne().reshape((1, -1))
  traject = sim_ode_proj_firstord(prob_inst, t_len)
  
  diff = traject - sol
  diff_norm = np.linalg.norm(diff, axis=1, ord=2)
  print(diff_norm)
  
  
if __name__ == '__main__':
  main()
