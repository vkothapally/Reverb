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
import gpuRIR

'''
class Room:
    
    def __init__(self):
        self.corners = np.array([[0,0], [0,12], [12,12], [12,0]]).T
        self.fs=8000
        self.max_order=8
        self.absorption=0.2
        self.room = pra.Room.from_corners(self.corners, fs=self.fs, max_order=self.max_order, absorption=self.absorption)
        self.room.extrude(2.)
        
    def show(self):
        fig, ax = self.room.plot(img_order=3)
        
    def plotRIR(self):
        self.room.plot_rir()
    
    def source(self, audiofile, location):
        signal, fs = sf.read(audiofile)
        self.source_location = location
        self.room.add_source(self.source_location, signal=signal)
        
    def microphones(self, nmics, maxdistance):
        micD= maxdistance*(1-np.exp(-1*np.geomspace(0.03, 10.0, nmics+2)))
        micD = micD[1:-1]
        R = np.array([list(self.source_location[0]*np.ones(nmics)+micD), 
                      list(self.source_location[1]*np.ones(nmics)), 
                      list(self.source_location[2]*np.ones(nmics))]) 
        self.room.add_microphone_array(pra.MicrophoneArray(R, self.room.fs))
        
    def simulate(self):
        self.room.image_source_model(use_libroom=True)
        self.room.simulate()
        return self.room.mic_array.signals
    
    def getRoomInfo(self):
        return self.room
        
        

room = Room()
room.source(audiofile='Test.wav', location=[1., 1., 0.5])
room.microphones(nmics=6, maxdistance=10)
room.plotRIR()




class Rooms:
    
    def __init__(self, nrooms, roomsize):
        self.fs=8000
        self.rooms = self.build(nrooms, roomsize)
    
    def build(self, nrooms, roosize='all'):
        length = np.round(np.geomspace(6.0, 25.0, 5))
        width = np.round(np.geomspace(3.0, 12.0, 5))
        height = np.round([3.0,6.0])
        
        
        
    def show(self):
        fig, ax = self.room.plot(img_order=3)
        
    def plotRIR(self):
        self.room.plot_rir()
    
    def source(self, audiofile, location):
        signal, fs = sf.read(audiofile)
        self.source_location = location
        self.room.add_source(self.source_location, signal=signal)
        
    def microphones(self, nmics, maxdistance):
        micD= maxdistance*(1-np.exp(-1*np.geomspace(0.03, 10.0, nmics+2)))
        micD = micD[1:-1]
        R = np.array([list(self.source_location[0]*np.ones(nmics)+micD), 
                      list(self.source_location[1]*np.ones(nmics)), 
                      list(self.source_location[2]*np.ones(nmics))]) 
        self.room.add_microphone_array(pra.MicrophoneArray(R, self.room.fs))
        
    def simulate(self):
        self.room.image_source_model(use_libroom=True)
        self.room.simulate()
        return self.room.mic_array.signals
    
    def getRoomInfo(self):
        return self.room

room_sz = [3,3,2.5]  # Size of the room [m]
nb_src = 2  # Number of sources
pos_src = np.array([[1,2.9,0.5],[1,2,0.5]]) # Positions of the sources ([m]
nb_rcv = 3 # Number of receivers
pos_rcv = np.array([[0.5,1,0.5],[1,1,0.5],[1.5,1,0.5]])	 # Position of the receivers [m]
orV_rcv = np.matlib.repmat(np.array([0,1,0]), nb_rcv, 1) # Vectors pointing in the same direction than the receivers
mic_pattern = "card" # Receiver polar pattern
abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
T60 = 3.0	 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]
fs=16000.0 # Sampling frequency [Hz]

beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

t = np.arange(int(ceil(Tmax * fs))) / fs
plt.plot(t, RIRs.reshape(nb_src*nb_rcv, -1).transpose())


'''



def build(self, nrooms, roosize='all'):
        length = np.round(np.geomspace(6.0, 25.0, 5))
        width = np.round(np.geomspace(3.0, 12.0, 5))
        height = np.round([3.0,6.0])













