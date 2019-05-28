#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:30:55 2019

@author: Vinay Kothapally

Room Acoustics Simulation 
"""

import os, glob
import numpy as np


roomInfo = []
for roomsize in ['small', 'medium', 'large']:
    with open( os.getcwd() + '/RIR Database/'+ roomsize + '/room_info') as f:
        data = f.readlines()
    [roomInfo.append(k.split(' ')) for k in data];
