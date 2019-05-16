#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:32:34 2019

@author: vinaykothapally
"""

import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from math import ceil
import numpy.matlib
import gpuRIR, itertools

class simRIR:
    
    def __init__(self, nrooms, nmics, maxdistance, nT60s, audiofile):
        self.rooms = self.build(nrooms)
        self.source(audiofile)
        self.microphones(nmics, maxdistance)
        self.reverbtime(nT60s)
        
         
    def build(self, nrooms):
        length = list(np.round(np.geomspace(10.0, 25.0, 20)))
        width = list(np.round(np.geomspace(6.0, 12.0, 10)))
        height = np.round([3.0,6.0])
        rooms = list(set(list(itertools.product(length, width, height))))
        rooms = sorted(rooms, key=lambda tup: tup[0])
        rooms = [list(rooms[int(k)]) for k in np.linspace(1,len(rooms)-1, nrooms)]
        print('*** Done Adding Room Information')
        return rooms
    
    def source(self, audiofile):
        self.nsrc = 1 
        self.signal, self.fs = sf.read(audiofile)
        self.source_location = np.array([1., 1., 0.5])
        print('*** Done Addding Source Location')
        
    def microphones(self, nmics, maxdistance):
        self.nmics = nmics
        self.mic_pattern = "omni"
        micD= maxdistance*(1-np.exp(-1*np.geomspace(0.03, 10.0, nmics+2)))
        micD = micD[1:-1]
        self.microphone_location = np.array([list(self.source_location[0]*np.ones(nmics)+micD), 
                                             list(self.source_location[1]*np.ones(nmics)), 
                                             list(self.source_location[2]*np.ones(nmics))]) 
        print('*** Done Adding Mictrophones Locations')
        
    def reverbtime(self, nT60s):
        self.T60s = np.geomspace(50e-3, 10, nT60s)
        print('*** Done Adding T60 Information!')
        
    
    def simulate(self):
        orV_rcv = np.matlib.repmat(np.array([0,0]), self.nmics, 1) # Vectors pointing in the same direction than the receivers
        abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
        att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
        att_max = 60.0 # Attenuation at the end of the simulation [dB]
        
        RIRs = {}
        room_count = 0
        for room_sz in self.rooms:
            room_count = room_count + 1
            RIRs['Room'+str(room_count)] = {}
            T60_count = 0
            for T60 in self.T60s:
                T60_count = T60_count + 1
                RIRs['Room'+str(room_count)]['T60_'+str(T60_count)] = {}
                mic_count = 0
                for pos_rcv in self.microphone_location:
                    mic_count = mic_count+1
                    beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
                    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
                    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
                    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
                    RIRs['Room'+str(room_count)]['T60_'+str(T60_count)]['mic'+str(mic_count)] = gpuRIR.simulateRIR(room_sz, beta, self.source_location, pos_rcv, nb_img, Tmax, self.fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=self.mic_pattern)
        return RIRs




Rooms = simRIR(nrooms=4, nmics=1, maxdistance=3, nT60s=2, audiofile='Test.wav')

temp = Rooms.simulate()




