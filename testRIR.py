#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:48:33 2019

@author: vinaykothapally

Script to Check the Room Impulse Response Simulations are Good Enough
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt


RoomInfo = np.load('./RIR_Database/Simulated/RoomInfo.npy')

test = pd.DataFrame(RoomInfo)
for i, row in test.iterrows():
    Room_dimensions = row[2]
    Array_Info = row[9]
    break


def plotArray(Array_Info):
    for arrays in Array_Info:
        array_loc = np.array(np.vstack([Array_Info[arrays][k] for k in Array_Info[arrays]]))
        print(array_loc)
        plt.scatter(array_loc[:,0], array_loc[:,1])
        return array_loc
        
#array_loc = plotArray(Array_Info)