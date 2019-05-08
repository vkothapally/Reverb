#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:32:34 2019

@author: vinaykothapally
"""

import numpy as np
import soundfile as sf
import pyroomacoustics as pra


class Room:
    
    def __init__(self):
        self.corners = np.array([[0,0], [0,6], [6,6], [6,0]]).T
        self.fs=8000
        self.max_order=8
        self.absorption=0.2
        self.room = pra.Room.from_corners(self.corners, fs=self.fs, max_order=self.max_order, absorption=self.absorption)
        self.room.extrude(2.)
        
    def show(self):
        fig, ax = self.room.plot(img_order=3)
        
    def plotRIR(self):
        self.room.plot_rir()

        

    def source(self):
        signal, fs = sf.read("Test.wav")
        self.room.add_source([1., 1., 0.5], signal=signal)


    def microphones(self):
        R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])  # [[x], [y], [z]]
        self.room.add_microphone_array(pra.MicrophoneArray(R, self.room.fs))
        
    def simulate(self):
        self.room.image_source_model(use_libroom=True)
        self.room.simulate()
        return self.room.mic_array.signals




room = Room()
room.microphones()
room.source()
signals = room.simulate()
