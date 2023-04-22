# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Times New Roman",
  "figure.dpi": 300,
  "font.size": 12,
  "lines.linewidth": 2,
  "font.weight": "bold",
  "axes.labelweight": "bold",
})


font_size = 14
linewidth = 1.5
save_path = './figs/nash_cournot_ode.png'
pt_cnt = int(5e3)
t_len = 1000

pcolor = ['#225B70', '#1693B0', '#F05A5B', '#C0C0C0', '#FEC015', '#e2aaa7', '#8EB900']
linedash = ['--', ':', '--', '--', ':', '-.']
linemarkers = ['o', 'v', 'v', 'v', 'o', '*', ]


# Define a function to plotting with better markers
def marker(T, n, mtype='linear'):
  marker = []
  
  if mtype == 'log':
    for k in range(n):
      temp = int(np.round(T ** (k / n))) - 1
      if temp <= 1:
        temp = 1
      if temp not in marker:
        marker.append(temp)
  elif mtype == 'linear':
    marker = list(range(1, T + 1, T // n))
  
  return marker


Time_marker = marker(pt_cnt, 10, 'log')

# ----- prepare the data for relative distances and potential differences -----
file_strs = {
  'proj1': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/proj_0.2_0.2.pkl',
  'proj2': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/proj_0.5_0.5.pkl',
  'proj3': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/proj_1_1.pkl',
  'yosida1': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/yosida_10_10.pkl',
  'yosida2': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/yosida_20_20.pkl',
  'fbf1': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/fbf_0.1_0.1_0.5.pkl',
  'fbf2': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/fbf_0.1_0.02_0.5.pkl',
  'fbf3': '/Users/murphyhuang/dev/src/github.com/murphyyhuang/odes-monoton-inclu/pkl/fbf_0.01_0.02_0.5.pkl',
}

legends = {
  'proj1': r'$\lambda = 0.2, \alpha = 0.2$',
  'proj2': r'$\lambda = 0.5, \alpha = 0.5$',
  'proj3': r'$\lambda = 1, \alpha = 1$',
  'yosida1': r'$\alpha = 10, \epsilon = 10$',
  'yosida2': r'$\alpha = 20, \epsilon = 20$',
  'fbf1': r'$\epsilon = 0.1, \gamma(t) = 0.02 / t^{0.5}$',
  'fbf2': r'$\epsilon = 0.01, \gamma(t) = 0.02 / t^{0.5}$',
  'fbf3': r'$\epsilon = 0.1, \gamma(t) = 0.1 / t^{0.5}$',
}


def read_data():
  
  df_res = pd.DataFrame()
  for alg_key, file_str in file_strs.items():
    with open(file_str, 'rb') as pkl_reader:
      result_tmp_dict = pickle.load(pkl_reader)
      # relative distances - original data
      relative_dist_res = result_tmp_dict['diff_norm']
      df_res[alg_key + '_diff_norm'] = relative_dist_res
  
  df_plot = pd.DataFrame({
    'Time_1': list(np.linspace(t_len/pt_cnt, t_len, pt_cnt)),
    'Time_2': list(np.linspace(t_len/pt_cnt + 10, t_len + 10, pt_cnt)),
  })
  df_plot = pd.concat([df_plot, df_res], ignore_index=False, axis=1)
  
  return df_plot


def plot_res():
  
  df_plot = read_data()
  
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

  for ienum, iaxe in enumerate(axes):
    iaxe.set_xlabel('Time ' + r'$t$', fontsize=font_size)
    iaxe.grid(color='black', linestyle='--', linewidth=0.1)

  # ------- subplot(a): projected first-order dynamics  -------
  axes[0].set_title('(a) Performance of the projected 1st-order dynamics')
  
  for choice_idx, plot_key in enumerate(['proj1', 'proj2', 'proj3']):
  
    axes[0].loglog(
      df_plot['Time_1'], df_plot[plot_key + '_diff_norm'],
      linestyle=linedash[choice_idx % len(linedash)],
      marker=linemarkers[choice_idx % len(linemarkers)],
      markevery=Time_marker, color=pcolor[choice_idx % len(pcolor)],
      linewidth=linewidth, markersize=5, label=legends[plot_key],
    )
  axes[0].legend()

  # ------- subplot(b): second-order Yosida regularization dynamics  -------
  axes[1].set_title('(b) Performance of the 2nd-order dynamics')

  for choice_idx, plot_key in enumerate(['yosida1', 'yosida2', ]):
    axes[1].loglog(
      df_plot['Time_2'], df_plot[plot_key + '_diff_norm'],
      linestyle=linedash[choice_idx % len(linedash)],
      marker=linemarkers[choice_idx % len(linemarkers)],
      markevery=Time_marker, color=pcolor[choice_idx % len(pcolor)],
      linewidth=linewidth, markersize=5, label=legends[plot_key],
    )
  axes[1].legend()
    
  # ------- subplot(c): fbf dynamics  -------
  axes[2].set_title('(c) Performance of the FBF dynamics')

  for choice_idx, plot_key in enumerate(['fbf1', 'fbf2', 'fbf3', ]):
    axes[2].loglog(
      df_plot['Time_1'], df_plot[plot_key + '_diff_norm'],
      linestyle=linedash[choice_idx % len(linedash)],
      marker=linemarkers[choice_idx % len(linemarkers)],
      markevery=Time_marker, color=pcolor[choice_idx % len(pcolor)],
      linewidth=linewidth, markersize=5, label=legends[plot_key],
    )
  axes[2].legend()
    
  plt.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')


def main():
  plot_res()


if __name__ == '__main__':
  main()