#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:38:46 2019

@author: vinay
"""
import os
import glob, shutil
import soundfile as sf 
import numpy as np
import pandas as pd
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats




def f0_estimate(infile, minf0, maxf0):
    outfile = infile.split('.')[0].replace('timit', 'timit_processed')+'.f0'
    command = '/home/vinay/Documents/Tools/GitHub/REAPER/build/reaper -t -m '+ \
              str(minf0) +' -x ' + str(maxf0) +' -i '+infile+' -f '+outfile+' -a'
    os.system(command)
    return

def sampleData(dataset, nSamples):
    if np.mod(nSamples,2) is 0:
        mSamples = int(nSamples/2)
        fSamples = int(nSamples/2)
    else:
        mSamples = int(nSamples/2 - 0.5)
        fSamples = int(nSamples/2 + 0.5)
    male = trainData[trainData['Gender']=='M'].sample(n=mSamples, random_state=1)
    female = trainData[trainData['Gender']=='F'].sample(n=fSamples, random_state=2)
    sampledData = pd.concat([male, female])
    sampledData = sampledData.reset_index(drop=True)
    return sampledData


def timit_dataframe(dataset, setaudiofmt, estimateF0):
    train_files = {'Filename':[], 'Speaker':[], 'Gender':[], 
                   'AudioPath':[], 'TransPath':[], 'WordPath':[], 
                   'PhonemePath':[], 'F0 Estimate':[]}
    for file in glob.glob(dataset+'/*/*/*.wav', recursive=True):
        filename = '-'.join(file.split('/')[5:]).split('.')[0]
        train_files['Filename'].append(filename)
        train_files['Speaker'].append(filename.split('-')[3][1:])
        train_files['Gender'].append(filename.split('-')[3][0].upper())
        train_files['AudioPath'].append(file)
        train_files['TransPath'].append(file.split('.')[0]+'.txt')
        train_files['WordPath'].append(file.split('.')[0]+'.wrd')
        train_files['PhonemePath'].append(file.split('.')[0].replace('timit', 'timit_processed')+'.phn')
        train_files['F0 Estimate'].append(file.split('.')[0].replace('timit', 'timit_processed')+'.f0')
        if setaudiofmt:
            audio, fs = sf.read(file)
            sf.write(file, audio, fs)
        if estimateF0:
            if filename.split('-')[3][0].upper() == 'M':
                f0_estimate(file, minf0=70, maxf0=160)
            else:
                f0_estimate(file, minf0=120, maxf0=300)
        
    return pd.DataFrame(train_files)


def getRIRfiles(RIRDataLoc):
    RIRfiles = glob.glob(RIRDataLoc+'/*.wav')
    RIRData = {'RIRName':[], 'RIRLoc':[], 'Room':[], 'T60':[], 'Array':[], 'Mic':[]}
    for file in RIRfiles:
        fileLoc = file
        fileName = os.path.basename(fileLoc)
        RIRData['RIRName'].append(fileName)
        RIRData['RIRLoc'].append(fileLoc)
        RIRData['Room'].append(int(fileName.split('_')[1]))
        RIRData['T60'].append(int(fileName.split('_')[3]))
        RIRData['Array'].append(int(fileName.split('_')[5]))
        RIRData['Mic'].append(int(fileName.split('_')[7].split('.')[0]))
    
    RIRData = pd.DataFrame(RIRData).sort_values(['RIRName'])
    RIRData = RIRData.reset_index(drop=True)
    return RIRData

def createFolderStructure(RIR_Database, destLoc):
    if os.path.isdir(destLoc):
        shutil.rmtree(destLoc)
    os.mkdir(destLoc)
    for roomNum in RIR_Database['Room'].unique():
        os.mkdir(destLoc+'/Room_'+str(roomNum))
        for rtNum in RIR_Database['T60'].unique():
            os.mkdir(destLoc+'/Room_'+str(roomNum)+'/RT_'+str(rtNum))
            for arrayNum in RIR_Database['Array'].unique():
                os.mkdir(destLoc+'/Room_'+str(roomNum)+'/RT_'+str(rtNum)+'/Array_'+str(arrayNum))
    
    return

def simTimitReverb(smallData, RIR_Database, destLoc):
    createFolderStructure(RIR_Database, destLoc)
    
    
    
    
    return




# ++++++++++++++++ MAIN FUNCTION CALLS +++++++++++++++++++++++++++#
    
trainData = timit_dataframe('/home/vinay/Documents/Timit/timit/train', setaudiofmt=0, estimateF0=0)
smallData = sampleData(trainData, nSamples=100)
RIR_Database = getRIRfiles('/home/vinay/GitHub/Reverb/RIR_Database/Simulated')
simTimitReverb(smallData, RIR_Database, '/home/vinay/Documents/Timit/Reverb')












































































