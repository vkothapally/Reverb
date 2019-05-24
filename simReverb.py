#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:32:34 2019

@author: vinaykothapally
"""

import os
#import gpuRIR
import itertools
import numpy as np
import pandas as pd
import numpy.matlib
import soundfile as sf
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class simRIR:
    
    def __init__(self, nrooms, narrays, maxdistance, nT60s):
        self.rooms = self.build(nrooms)
        self.fs = 16000
        self.source()
        self.reverbtime(nT60s)
        self.microphones(narrays, maxdistance)
        self.getRIRs_test()
        
         
    def build(self, nrooms):
        length = list(np.round(np.geomspace(5, 30.0, 20)))
        width = list(np.round(np.geomspace(3.0, 14.0, 15)))
        height = np.round([2.5,4.0,6.0])
        rooms = {}
        rooms_options = list(set(list(itertools.product(length, width, height))))
        rooms_options = sorted(rooms_options, key=lambda tup: tup[0])
        rooms_options = [rooms_options[int(k)] for k in np.linspace(1,len(rooms_options)-1, nrooms)]
        for k in range(len(rooms_options)):
            rooms[k] = {}
            rooms[k]['Dimensions'] = rooms_options[k]
            rooms[k]['Area (Sq.ft)'] = rooms_options[k][0]*rooms_options[k][1]*10.7639
            rooms[k]['Size'] = (rooms[k]['Area (Sq.ft)']<1000)*'Small' + \
                               (rooms[k]['Area (Sq.ft)']>1000 and rooms[k]['Area (Sq.ft)']<2000)*'Medium' + \
                               (rooms[k]['Area (Sq.ft)']>2000)*'Large'
            rooms[k]['MaxDistance'] = np.round(np.sqrt((rooms_options[k][0]-1)**2 + (rooms_options[k][1]-1)**2)-3, decimals=2) 
        rooms = pd.DataFrame(rooms).transpose()
        print('*** Done Adding Room Information')
        return rooms
    
    def source(self):
        self.nsrc = 1 
        source_location = len(self.rooms)*[[1., 1., 0.5]]
        self.rooms['Source'] = source_location
        print('*** Done Addding Source Location')
        
    def microphones(self, narrays, maxdistance):
        self.rooms['Mic Pattern'] = len(self.rooms)*['omni']
        micD= np.round(maxdistance*(1-np.exp(-1*np.geomspace(0.5, maxdistance, narrays+2))),decimals=3)
        micD = [1,3,6]
        self.rooms['nArrays'] = ""
        self.rooms['nMics'] = ""
        self.rooms['Array Location'] = ""
        self.rooms['Array Distance'] = ""
        self.rooms['T60'] = ""
        for i, row in self.rooms.iterrows():
            self.rooms['nArrays'][i] =  int(np.sum(1.0*(micD <= self.rooms['MaxDistance'][i])))
            self.rooms['nMics'][i] =  5*int(np.sum(1.0*(micD <= self.rooms['MaxDistance'][i])))
            self.rooms['Array Distance'][i] = list(micD[0:self.rooms['nArrays'][i]])
            self.rooms['Array Location'][i] = {} 
            self.rooms['T60'][i] = self.T60s
            for k in range(self.rooms['nArrays'][i]):
                angle = np.arctan((self.rooms['Dimensions'][i][1]-1)/(self.rooms['Dimensions'][i][0]-1))
                array_center = [np.round(self.rooms['Source'][i][0]+micD[k]*np.cos(angle),decimals=2), 
                                                                       np.round(self.rooms['Source'][i][1]+micD[k]*np.sin(angle),decimals=2), 
                                                                       self.rooms['Source'][i][2]]
                self.rooms['Array Location'][i]['Array_'+str(k)] = self.array_points(array_center, micD[k], angle)
                    #if any(np.array(self.rooms['Array Location'][i]['Array-'+str(k)]['Mic-'+str(m)]) >= np.asarray(self.rooms['Dimensions'][i])):
                    #    print('Microphone outside room! Check Logic--- Room: '+str(i)+'--  Array-'+str(k)+ '--  Mic-'+str(m))
            
        print('*** Done Adding Mictrophones Locations')
        
    def array_points(self, center, d, theta):
        array = {}; a = 2e-2;
        array['Mic_0'] = list(np.round(np.array(center) + np.array([d*np.cos(theta)+a*np.sin(theta),d*np.sin(theta)-a*np.cos(theta),0]), decimals=2))
        array['Mic_1'] = list(np.round(np.array(center) + np.array([d*np.cos(theta),d*np.sin(theta),0]), decimals=2))
        array['Mic_2'] = list(np.round(np.array(center) + np.array([d*np.cos(theta)-a*np.sin(theta),d*np.sin(theta)+a*np.cos(theta),0]), decimals=2))
        array['Mic_3'] = list(np.round(np.array(center) + np.array([(d-a)*np.cos(theta),(d-a)*np.sin(theta),0]), decimals=2))
        array['Mic_4'] = list(np.round(np.array(center) + np.array([(d+a)*np.cos(theta),(d+a)*np.sin(theta),0]), decimals=2))
        return array

        
    def reverbtime(self, nT60s):
        self.T60s = list(np.round(np.geomspace(100e-3, 10, nT60s), decimals=3))
        print('*** Done Adding T60 Information!')
    
    def getRIRs_test(self):
        print('Starting Simulations')
        self.rooms['RIR'] = ""
        for i, row in tqdm(self.rooms.iterrows(), desc='*** Simulating RIRs'):
            self.rooms['RIR'][i] = {}
            room_sz = list(self.rooms['Dimensions'][i])
            pos_src =  np.array([self.rooms['Source'][i]])
            for k in range(len(self.T60s)):
                self.rooms['RIR'][i][k] = {}
                self.rooms['RIR'][i][k]['T60'] = self.T60s[k]
                for array in self.rooms['Array Location'][i]:
                    self.rooms['RIR'][i][k][array] = {}
                    for mic in self.rooms['Array Location'][i][array]:
                        pos_rcv = np.array([self.rooms['Array Location'][i][array][mic]])
                        self.rooms['RIR'][i][k][array][mic] = 'Room_'+str(i)+'_RT_'+str(k)+'_'+array+'_'+mic+'.wav'
                        beta = gpuRIR.beta_SabineEstimation(room_sz, self.T60s[k], abs_weights=[0.9]*5+[0.5])
                        Tdiff= gpuRIR.att2t_SabineEstimator(15, self.T60s[k])
                        nb_img = gpuRIR.t2n(Tdiff, room_sz)
                        h =  gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, 
                                                nb_img, 10, 
                                                self.fs, 
                                                Tdiff=Tdiff, 
                                                orV_rcv=None, 
                                                mic_pattern=self.rooms['Mic Pattern'][i]).flatten()
                        sf.write(os.getcwd()+'/RIR_Database/Simulated/'+self.rooms['RIR'][i][k][array][mic],h,self.fs)
        print('*** Done Simulating RIRs')
        numpy.save(os.getcwd()+'/RIR_Database/Simulated/RoomInfo', self.rooms, allow_pickle=True, fix_imports=True)
        return
                 

Rooms = simRIR(nrooms=30, narrays=5, maxdistance=8, nT60s=6)
RoomDatabase = Rooms.rooms



