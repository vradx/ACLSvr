# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:37:12 2023

@author: VALERIO RADICCHI

@email: valerio.radicchi@edu.fh-kaernten.ac.at
"""

from datetime import datetime
import fnmatch
from joblib import dump
import json
import numpy as np
import os
import pandas as pd
import pydub
import scipy
import shutil
import soundfile as sf
import sys


def getEventsFromJSON(jsonDir, jsonFile):
    
    jsonFileName = os.path.join(jsonDir, jsonFile)  # Setting the full-filename
        
    with open(jsonFileName) as json_file:
        jasonDict = json.load(json_file)     # Loading the .json file into a Dictionary variable
    
    event_data = jasonDict['event_annotation']  # Selecting only 'event_annotation' section data from .json file
    if len(event_data) > 0:     # Event data are present
        eventDF = pd.DataFrame.from_dict(event_data, orient='columns')

    else:                       # Event data are missing (probably the record quality was poor)
        data = {'start':[0], 'end':[0], 'type':['none']}  # Simulating a void event record for output compatibiity
        eventDF = pd.DataFrame(data)
    
    eventDF['start'] = pd.to_numeric(eventDF['start'])  # Converting 'start' value to integer
    eventDF['end'] = pd.to_numeric(eventDF['end'])      # Converting 'end' value to integer
    eventDF.sort_values(by=['start'], inplace=True)
    eventDF.reset_index(drop=True, inplace=True)
    return eventDF

def equalizeEventLength(eventFolder, targetDuration, padding):
    audioDataEqz = np.array([])
    eventFiles = os.listdir(eventFolder)
    for fileName in eventFiles:
        if not os.path.isdir(os.path.join(eventFolder, fileName)):
            try:
                eventFileName = os.path.join(eventFolder, fileName)
                audioData, Fs = sf.read(eventFileName)
                if padding == "M":
                    audioDataMean = np.mean(audioData)
                diffMsec = targetDuration - len(audioData)*1000/Fs
                diffSamples = diffMsec * Fs / 1000
                if diffSamples % 2 != 0:
                    if padding == "Z":
                        head = np.zeros(int(diffSamples/2))
                        tail = np.zeros(int(diffSamples/2)+1)
                    else:
                        head = np.full(int(diffSamples/2), audioDataMean)
                        tail = np.full((int(diffSamples/2)+1), audioDataMean)
                else:
                    if padding == "Z":
                        head = np.zeros(int(diffSamples/2))
                        tail = np.zeros(int(diffSamples/2))
                    else:
                        head = np.full(int(diffSamples/2), audioDataMean)
                        tail = np.full(int(diffSamples/2), audioDataMean)
                audioDataEqz = np.concatenate((head, audioData, tail))
                sf.write(eventFileName, audioDataEqz, Fs)
            except Exception as e:
                print('Failed to equalize %s\nReason: %s' % (eventFileName, e))
    
    return len(audioDataEqz)

def getInfoFromFilename(filename, filteredYN):
    mainSplit = filename.split(sep='-')
    eventRec = mainSplit[0]
    eventRec = eventRec.split(sep='_')[-1]   # Gets last component (rec. number) from filename
    eventNum = mainSplit[1]
    eventNum = eventNum.split(sep='_')[-1]
    if filteredYN == "Y":
        eventType = mainSplit[-2]
    else:
        eventType = mainSplit[-1]
        eventType = eventType.split(sep='.')[0]
    match eventType:
        case "Normal":          # Assigned event label is "Normal"
            eventLabel = 1
        case "Wheeze":          # Assigned event label is "Wheeze"
            eventLabel = 2
        case "Fine Crackle":    # Assigned event label is "Fine Crackle"
            eventLabel = 3
        case "Coarse Crackle":  # Assigned event label is "Coarse Crackle"
            eventLabel = 4
        case "Wheeze+Crackle":  # Assigned event label is "Wheeze+Crackle"
            eventLabel = 5
        case "Rhonchi":  		# Assigned event label is "Rhonchi"
            eventLabel = 6
        case "Stridor":  		# Assigned event label is "Stridor"
            eventLabel = 7
        case _:             # No any event label was assigned
            eventLabel = 0
            
    return eventRec, eventNum, eventLabel


def getWavFilesFigures(wavFilesNumber):
    if wavFilesNumber < 10:
        wavFilesFigures = 1
    elif wavFilesNumber < 100:
        wavFilesFigures = 2
    elif wavFilesNumber < 1000:
        wavFilesFigures = 3
    elif wavFilesNumber < 10000:
        wavFilesFigures = 4
    elif wavFilesNumber < 100000:
        wavFilesFigures = 5
    else:
        wavFilesFigures = 9
    
    return wavFilesFigures
    

def cleanDataFolder(folder, pattern):
    for file in os.listdir(folder):   # Cleaning the given folder from old 'pattern' files
        if fnmatch.fnmatch(file, pattern):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def dumpOutDataMatrix(outData, k, numFigures):
    segFileName = 'segmentedData_' + str(k).zfill(numFigures) + '.bin.z'  # Building dump file name
    segDataFile = os.path.join(runDir, segFileName)
    with open(segDataFile, 'wb') as f:      # Opening dump file for writing in binary mode
        dump(outData, f, compress='zlib')   # Dumping data by using the joblib.dump() method in compressed mode


def getEventFileShape(eventFolder, pattern):
    eventFiles = os.listdir(eventFolder)
    for fileName in eventFiles:
        if fnmatch.fnmatch(fileName, pattern):
            eventFileName = os.path.join(eventFolder, fileName)
            f = sf.SoundFile(eventFileName)
            break
    
    return f.frames


def buildEventMatrix(eventFolder, pattern, audioDataLen, labelFilter, filtered):
    eventFiles = os.listdir(eventFolder)
    eventFileNum = len(fnmatch.filter(eventFiles, pattern))
    dumpFileFigures = getWavFilesFigures(eventFileNum)
    outData = np.zeros(shape=(1,audioDataLen+3), dtype=float)
    k = 0
    for fileName in eventFiles:
        if fnmatch.fnmatch(fileName, pattern):
            eventFileName = os.path.join(eventFolder, fileName)
            try:
                tmpData = np.zeros(shape=(1,audioDataLen+3), dtype=float)
                eventRec, eventNum, eventLabel = getInfoFromFilename(fileName, filtered)
                if (eventLabel in labelFilter) or labelFilter[0] == 0:
                    labelFound = True
                    audioData, Fs = sf.read(eventFileName)
                    tmpData[0,0] = eventRec
                    tmpData[0,1] = eventNum
                    tmpData[0,2] = eventLabel
                    tmpData[0,3:] = audioData
    
                    if k==0 or k % MAXFILES == 0:
                        outData = tmpData
                    else:
                        outData = np.vstack((outData, tmpData))
                    k += 1
                    if k == 1099:
                        pass
                else:
                    labelFound = False
            except Exception as e:
                print('Failed to process %s\nReason: %s' % (eventFileName, e))
            
            if k % MAXFILES == 0 and labelFound:
                dumpOutDataMatrix(outData, k, dumpFileFigures)
                outData = np.zeros(shape=(1,audioDataLen+3), dtype=float)
    dumpOutDataMatrix(outData, k, dumpFileFigures)
    
    return k


def getAudioFileLoudness(fullFileName):
    
    sound = pydub.AudioSegment.from_file(fullFileName, "wav")  # Importing audio file from disk
    audioLoudness = sound.dBFS
    
    return audioLoudness


def match_target_amplitude(sound, target_dBFS):
    delta_dBFS = target_dBFS - sound.dBFS  # Calculating the amplification/attenuation factor to be applied to the wav
    
    return sound.apply_gain(delta_dBFS)


def butterFilter(order, cutoff, ftype, sampling, audio):
    # Getting Second Order Section (SOS) filtering parameters
    sos = scipy.signal.butter(order, cutoff, btype=ftype, fs=sampling, output='sos')
    fltAudio = scipy.signal.sosfilt(sos, audio)  # Applying filter to audio signal
    
    return fltAudio


def filterEventFiles(folderName, targetDir, pattern):
    
    audioFiles = []
    wavFiles = os.listdir(folderName)
    for fileName in wavFiles:
        if fnmatch.fnmatch(fileName, pattern):
            audioFiles.append(fileName)     # Appending matching files to the audioFiles list
    audioFilesArr = np.array(audioFiles).T  # Converting (transposed) audioFiles list to numpy array
    for k in range(0, len(audioFilesArr)):      # Cycling through the audioFiles array
        fName, fExt = os.path.splitext(audioFilesArr[k])
        outFileName = fName + '-FILT' + fExt  # Preparing output file name
        origFileName = os.path.join(folderName, fName + fExt)  # Input full-file name
        filteredFileName = os.path.join(targetDir, outFileName)  # Output full-file name
        audioData, Fs = sf.read(origFileName) # Reading input wav file into a numpy array
        xF = butterFilter(BUTTER_ORDER, BUTTER_CUTOFF, 'highpass', Fs, audioData)                          
        sf.write(filteredFileName, xF, Fs)  # Writing (temporary) filtered file to disk


def calculateMeanLoudness(eventFolder, eventArray):
    
    eventsLoudness = []
    for k in range(0, len(eventArray)):
        eventFullFile = os.path.join(eventFolder, eventArray[k])
        eventsLoudness.append(getAudioFileLoudness(eventFullFile))

    return np.average(eventsLoudness)


def normalizeLoudness(folderName, targetLoudness, pattern, normType):
       
    audioFiles = []
    wavFiles = os.listdir(folderName)
    for fileName in wavFiles:
        if fnmatch.fnmatch(fileName, pattern):
            audioFiles.append(fileName)     # Appending matching files to the audioFiles list
    audioFilesArr = np.array(audioFiles).T  # Converting (transposed) audioFiles list to numpy array
    if normType == "M":
        targetLoudness = calculateMeanLoudness(folderName, audioFilesArr)
        print('\nThe mean loudness is:', targetLoudness)
    for k in range(0, len(audioFilesArr)):      # Cycling through the audioFiles array
        fName, fExt = os.path.splitext(audioFilesArr[k])
        origFileName = os.path.join(folderName, fName + fExt)  # Input full-file name
        normFileName = origFileName  # Output full-file name
        sound = pydub.AudioSegment.from_file(origFileName, "wav")  # Re-importing filtered file from disk
        
        ### WRITE ON EXTERNAL FILE THE ARRAY OF ORIGINAL AUDIO FILES LOUDNESS VALUES [filename, loudness]
        normalized_sound = match_target_amplitude(sound, targetLoudness)  # Normalizing filtered file
        normalized_sound.export(normFileName, format="wav")  # Exporting final output file to disk
        


# *** EXECUTION PARAMETERS *** 

BUTTER_ORDER = 20          # Order of the Butterworth filter
BUTTER_CUTOFF = 100        # Cut-off frequency of the Butterworth filter
DEF_PROCESSING = "N"       # "Y"= Re-Run full event extraction | "N"= Re-Use event audio files already available
DEF_MAX_DURATION = "N"     # "Y"= EQUALIZE AT MAX DURATION | "N"= EQUALIZE AT FIXED DURATION
DEF_MIN_LENGTH_MSEC = 0    # Min Milliseconds - TAKEN INTO ACCOUNT ONLY IF DEF_MAX_DURATION = "N"
DEF_MAX_LENGTH_MSEC = 1525 # Max Milliseconds - TAKEN INTO ACCOUNT ONLY IF DEF_MAX_DURATION = "N"
DEF_PADDING = "Z"          # "Z"= Padding by zeroes | "M"= padding by mean event value
DEF_FILTER = "N"           # "Y"= Apply Butter filter to event files | "N"= Do not filter event files
DEF_NORMALIZE = "Y"        # "Y"= Apply loudness normalization | "N"= Do not normalize event files
DEF_NORM_TYPE = "M"        # "M"= Use mean events loudness value | "F"= Use fixed loudness value
MAXFILES = 100
TARGET_LOUDNESS = -43      # Default loudness is -23 dB according to IBU standards

RUNTIME = "KL-N-A"  # "RM"= Rome       | "KL"= Klagenfurt
                    # "N" = Notebook   | "F" = Fixed
                    # "S" = Reduced DB | "A" = Full DB

if RUNTIME == "KL-N-A":
    wavDir = r'D:\Dataset\SPRSound-main\COMMON'
    jsonDir = r'D:\Dataset\SPRSound-main\COMMON'
    eventDir = r'D:\Dataset\SPRSound-main\COMMON\events'
    fltEventDir = r'D:\Dataset\SPRSound-main\COMMON\events\filtered'
    runDir = r'C:\Users\valer\Desktop\Thesis Software\_bin'
elif RUNTIME == "KL-N-S":
    wavDir = r'D:\Dataset\SPRSound-main\COMMON\Cherry\_BALANCED'
    jsonDir = r'D:\Dataset\SPRSound-main\COMMON\Cherry\_BALANCED'
    eventDir = r'D:\Dataset\SPRSound-main\COMMON\Cherry\_BALANCED\events'
    fltEventDir = r'D:\Dataset\SPRSound-main\COMMON\Cherry\_BALANCED\events\filtered'
    runDir = r'C:\Users\valer\Desktop\Thesis Software\_bin'
elif RUNTIME == "RM-F-S":
    wavDir = r'K:\Master\SJTU\SJTU-Respiratory_Sound_DB\Example'
    jsonDir = r'K:\Master\SJTU\SJTU-Respiratory_Sound_DB\Example'
    eventDir = r'K:\Master\SJTU\SJTU-Respiratory_Sound_DB\Example\events'
    fltEventDir = r'K:\Master\SJTU\SJTU-Respiratory_Sound_DB\Example\events\filtered'
    runDir = r'K:\MASTER\Run'
elif RUNTIME == "RM-N-S":
    wavDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\cherry'
    jsonDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\cherry'
    eventDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\cherry\events'
    fltEventDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\cherry\events\filtered'
    runDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\Run'
elif RUNTIME == "RM-N-A":
    wavDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\wav'
    jsonDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\json'
    eventDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\wav\events'
    fltEventDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\wav\events\filtered'
    runDir = r'E:\_MASTER_VALERIO_\SJTU-Respiratory_Sound_DB\Run'


fullProcessing = "?"

while fullProcessing.upper() != "Y" and fullProcessing.upper() != "N":
    fullProcessing = input("Do you want to re-run full event extraction from audiofiles ? [y/N] ") or DEF_PROCESSING
    fullProcessing = fullProcessing.upper()
    print("You entered: " + fullProcessing)
    if fullProcessing != "Y" and fullProcessing != "N":
        print("Valid inputs are: \"Y\" , \"N\"")

if fullProcessing == "Y":
    processingType = "FULL"
else:
    processingType = "MATRIX_ONLY"

if processingType == "FULL":
    eqzChoice = "?"
    eqzPadding = "?"

    while eqzChoice.upper() != "Y" and eqzChoice.upper() != "N":
        eqzChoice = input("Do you want to equalize events based on Max Duration ? [y/N] ") or DEF_MAX_DURATION
        eqzChoice = eqzChoice.upper()
        print("You entered: " + eqzChoice)
        if eqzChoice != "Y" and eqzChoice != "N":
            print("Valid inputs are: \"Y\" , \"N\"")
    
    if eqzChoice == "Y":
        eqzType = "MAX"
    else:
        eqzType = "FIX"
    
        eqzMinDuration = int(input("Enter target MIN duration (msec) for event equalization: [" + str(DEF_MIN_LENGTH_MSEC) + "] ")
                           or DEF_MIN_LENGTH_MSEC)
        eqzMaxDuration = int(input("Enter target MAX duration (msec) for event equalization: [" + str(DEF_MAX_LENGTH_MSEC) + "] ")
                           or DEF_MAX_LENGTH_MSEC)
        print("You entered:\n\tMin duration = " + str(eqzMinDuration) + " msec")
        print("\tMax duration = " + str(eqzMaxDuration) + " msec")
        
    while eqzPadding.upper() != "Z" and eqzPadding.upper() != "M":
        eqzPadding = input("Do you want to pad audio files by Zeros or by Mean event values ? [Z/m] ") or DEF_PADDING
        eqzPadding = eqzPadding.upper()
        print("You entered: " + eqzPadding)
        if eqzPadding != "Z" and eqzPadding != "M":
            print("Valid inputs are: \"Z\" , \"M\"")
        
filterOutdata = 'N'
selectedLabels = []

filterOutdata = (input("Do you want to filter outData matrix Labels [y/N]: ").upper() or filterOutdata)
print("You entered: " + filterOutdata)

if filterOutdata == 'Y':
    selectedLabels = input("Enter list of Labels target to be selected (separated by ','):" \
                           "[1=Normal, 2=Wheeze, 3=Fine Crackle, 4=Coarse Crackle, 5=Wheeze+Crackle, 6=Rhonchi, 7=Stridor]"\
                           "(Example: 1,3,4) : ")
    print("You entered: " + selectedLabels)
    selectedLabels = [int(x) for x in selectedLabels.split(",")]
else:
    selectedLabels = [0]

filterChoice = '?'

while filterChoice.upper() != "Y" and filterChoice.upper() != "N":
    filterChoice = input("Do you want to apply a Butterworth filter to audio events ? [y/N] ") or DEF_FILTER
    filterChoice = filterChoice.upper()
    print("You entered: " + filterChoice)
    if filterChoice != "Y" and filterChoice != "N":
        print("Valid inputs are: \"Y\" , \"N\"")

normalizeChoice = '?'

while normalizeChoice.upper() != "Y" and normalizeChoice.upper() != "N":
    normalizeChoice = input("Do you want to normalize event files loudness ? [Y/n] ") or DEF_NORMALIZE
    normalizeChoice = normalizeChoice.upper()
    print("You entered: " + normalizeChoice)
    if normalizeChoice != "Y" and normalizeChoice != "N":
        print("Valid inputs are: \"Y\" , \"N\"")

if normalizeChoice == "Y":
    eventsLoudness = []
    normalizationType = '?'
    while normalizationType.upper() != "F" and normalizationType.upper() != "M":
        normalizationType = input("Do you want to normalize files using fixed or mean loudness ? [f/M] ") or DEF_NORM_TYPE
        normalizationType = normalizationType.upper()
        print("You entered: " + normalizationType)
        if normalizationType == "F":
            targetLoudness = float(input("Enter target loudness (dB) for audio normalization: ")
                           or TARGET_LOUDNESS)
            print("You entered: " + str(targetLoudness))
        else:
            targetLoudness = TARGET_LOUDNESS    # Fake target loudness: will be overwritten later by Mean Loudness

print('Execution starting at:', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

w_pattern = '*.wav'   # Setting pattern for .wav files search
b_pattern = '*.bin.z' # Setting pattern for dump files file search

if processingType == "FULL":
    if not os.path.isdir(eventDir):
        os.makedirs(eventDir)        # Creating the './events' output folder if not existing
    else:
        for filename in os.listdir(eventDir):   # Cleaning the './events' folder from old .wav files
            if fnmatch.fnmatch(filename, w_pattern):
                file_path = os.path.join(eventDir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    audioFiles = []  # Initializing audioFiles list
    
    wavFiles = os.listdir(wavDir)
    for fileName in wavFiles:
        if fnmatch.fnmatch(fileName, w_pattern):
            audioFiles.append(fileName)     # Appending matching files to the audioFiles list
    
    audioFilesArr = np.array(audioFiles).T  # Converting (transposed) audioFiles list to numpy array
    if eqzType == "MAX":
        eventMaxLength = 0
    else:
        eventMinLength = eqzMinDuration
        eventMaxLength = eqzMaxDuration
    
    for k in range(0, len(audioFilesArr)):      # Cycling through the audioFiles array
        fullFileName = os.path.join(wavDir, audioFilesArr[k])  # Building input audio full-filename
    
        chunkLabelDF = getEventsFromJSON(jsonDir, audioFilesArr[k].replace('.wav', '.json'))
    
        audioData, Fs = sf.read(fullFileName)   # Reading input wav file into a numpy array
        
        for index, row in chunkLabelDF.iterrows():
            if row['type'] != 'none':
                eventFile = audioFilesArr[k].replace('.wav', '-') + 'Event_' + str(index) + '-' + row['type'] + '.wav'
                startSample = int(row['start'] * Fs / 1000)
                endSample = int(row['end'] * Fs / 1000)
                if eqzType == "MAX":
                    if int(row['end']) - int(row['start']) > eventMaxLength:
                        eventMaxLength = int(row['end']) - int(row['start'])
                    eventData = audioData[startSample:endSample]
                    eventFileName = os.path.join(eventDir, eventFile)
                    sf.write(eventFileName, eventData, Fs)
                elif eqzType == "FIX":
                    if (int(row['end']) - int(row['start']) >= eventMinLength) \
                        and \
                        (int(row['end']) - int(row['start']) <= eventMaxLength):    
                        eventData = audioData[startSample:endSample]
                        eventFileName = os.path.join(eventDir, eventFile)
                        sf.write(eventFileName, eventData, Fs)
    
    print('\nStarting event equalization...', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    
    audioDataShape = equalizeEventLength(eventDir, eventMaxLength, eqzPadding)
    
    if audioDataShape == 0:
        print('\nNo any valid event files have been found!')
        print('\nExiting program...')
        sys.exit()
    
# Applying Butter filter to event files
if filterChoice == "Y":
	print('\nApplying Butterworth filter to event files...', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
	if not os.path.isdir(fltEventDir):
		os.makedirs(fltEventDir)
	else:
		for file in os.listdir():
			os.remove(file)
	filterEventFiles(eventDir, fltEventDir, w_pattern)

# Applying normalization to event files
if normalizeChoice == "Y":
	print('\nApplying loudness normalization to event files...', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
	if filterChoice == "Y":    
		normalizeLoudness(fltEventDir, targetLoudness, w_pattern, normalizationType)
	else:
		normalizeLoudness(eventDir, targetLoudness, w_pattern, normalizationType)

print('\nBuilding event matrix...', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

cleanDataFolder(runDir, b_pattern)

if filterChoice == "Y":
    audioDataShape = getEventFileShape(fltEventDir, w_pattern)
    pickedEvents = buildEventMatrix(fltEventDir, w_pattern, audioDataShape, selectedLabels, "Y")
else:
    audioDataShape = getEventFileShape(eventDir, w_pattern)
    pickedEvents = buildEventMatrix(eventDir, w_pattern, audioDataShape, selectedLabels, "N")

if processingType == "FULL":
    print('\n' + str(len(audioFilesArr)) + ' audio files have been processed')
    print('\n' + str(pickedEvents) , 'events of types' , selectedLabels , 'have been selected from audio files')
    print('\nEvent max duration is:', str(eventMaxLength), 'msec')

print('\nExecution completed at:', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))



