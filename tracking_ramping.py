# -*- coding: utf-8 -*-
"""
Created on April 17
@author: LS276867
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
from optics_function import plot_twiss
from interpolation import calc_dip_coef,eval_horner
from track_function import track_cell,track,distribution_lost_turn,distribution_lost_turn_long
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize


file_seq = '/mnt/c/muco/code/6d_dq_5_sliced_ramp.json'
line=xt.Line.from_json(file_seq)
print('Loaded file')

