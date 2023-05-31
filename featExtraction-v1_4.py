import pandas as pd
import numpy as np
import datetime
import scipy
import fnmatch
import os
from joblib import load


def assembleOutDataMatrix(binFolder):
    pattern = '*.bin.z'  # Setting pattern for binary file search
    binFiles = os.listdir(binFolder)  # Loading a list of the files found in the search folder
    idx = 0

    for fileName in binFiles:  # Scanning the files list
        if fnmatch.fnmatch(fileName, pattern):
            binFullName = os.path.join(binFolder, fileName)  # The searched file has been found
            with open(binFullName, 'rb') as f:  # Opening input file for read in binary format
                outDataPart = load(f)  # Loading data by joblib.open() method
            if idx == 0:
                outData = outDataPart
            else:
                outData = np.vstack((outData, outDataPart))
            idx += 1
    return outData


def remove_rows(feature_df, max_majority_rows): # This function makes the classes even by random dropping the elements in excess
    label_counts = feature_df['Labels'].value_counts() # Count the Labels
    majority_label = label_counts.index[0] # identify the majority class label
    # majority_count = label_counts.iloc[0]
    # min_count = adventitious_class
    min_count = label_counts.min() # Extract the minority label numbers | Use for Binary classification |
    max_count = min_count + max_majority_rows # Define the max amount for the majority class | Use for Binary classification |
    # max_count = 2114
    
    drop_indices = [] # Prepare to remove indices and extract only numeric values
    for label in label_counts.index:
        count = label_counts[label]
        if label == majority_label:
            excess_count = count - max_count
            if excess_count > 0:
                drop_indices += feature_df[feature_df['Labels'] == label].sample(n=excess_count, random_state=5).index.tolist()
        else:
            if count > max_count:
                drop_indices += feature_df[feature_df['Labels'] == label].sample(n=count-max_count, random_state=5).index.tolist()
    return feature_df.drop(drop_indices)


# MAIN

binDir = "C:\\Users\\valer\\Desktop\\Thesis Software\\_bin\\"  # Loads all .bin from chosen folder
audioData = assembleOutDataMatrix(binDir)

# labels = audioData.iloc[:, 2]  # Get the labels from audioData
labels = audioData[:, 2]  # Get the labels from audioData
labels = labels.astype(int)

# Initialize the df
feature_df = pd.DataFrame(columns=['Variance', 'SMA_Coarse', 'Range',
                                   'Spectrum_Mean', 'SMA_Fine', 'ZCR', 'Spec_Kurtosis',
                                   'RMS', 'Labels'])

# Convert df to array
# audioData = audioData.to_numpy()

# Append labels
feature_df['Labels'] = labels

# Slice the array starting from 3rd column
audioData = audioData[:, 3:]
audioDataShape = np.shape(audioData)

# Iterate over each row of the dataframe
for i in range(0, audioDataShape[0]):
    
    # Calculate variance
    row_variance = audioData[i].var()
    feature_df.at[i, 'Variance'] = row_variance

    # Calculate the range
    row_max = max(audioData[i])
    row_min = min(audioData[i])
    row_range = abs(row_max - row_min)
    feature_df.at[i, 'Range'] = row_range

    # Calculate Simple Average (Coarse)
    sma = abs(np.mean(audioData[i]))
    feature_df.at[i, 'SMA_Coarse'] = sma

    # Calculate Zero-Crossing Rate
    zero_crossings = np.nonzero(np.diff(audioData[i] > 0))[0]
    zero_crossing_rate = (len(zero_crossings) / len(audioData[i])) * 1000
    feature_df.at[i, 'ZCR'] = zero_crossing_rate

    # Calculate Spectrum
    # spectrum = np.fft.fft(audioData[i]) # Returns complex numbers
    spectrum = abs(scipy.fft.dct(audioData[i]))  # Direct cosine transform
    feature_df.at[i, 'Spectrum_Mean'] = np.mean(spectrum)

    # Calculate Spectral Kurtosis
    spec_kurtosis = scipy.stats.kurtosis(spectrum)
    feature_df.at[i, 'Spec_Kurtosis'] = spec_kurtosis

    # Calculate Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.square(audioData[i])))
    feature_df.at[i, 'RMS'] = rms

    # Calculate Simple Average (fine)
    window_size = int(128)
    step_size = int(16)
    moving_average = []

    for j in range(0, len(audioData[i]), step_size):
        sample_window = audioData[i][j:j + window_size - 1]
        moving_average.append(abs(sample_window.mean()))
        if j > len(audioData[i]) - step_size:
            break

    smaF = np.array(moving_average).mean()
    feature_df.at[i, 'SMA_Fine'] = smaF

    if i == 0 or i % 1000 == 0:
        print("Iteration: ", i, "--> Time: ", datetime.datetime.now(), "\n")



value_counts = feature_df['Labels'].value_counts()
count_table = pd.DataFrame({'Label': value_counts.index, 'count': value_counts.values})
adventitious_class = int(count_table.iloc[-1, -1]) # minority class in binary classification
# adventitious_class = int(count_table.iloc[1, 1])

# Max majority rows is the max amount of rows that you want to have
feature_df = remove_rows(feature_df, max_majority_rows=0)

print(feature_df['Labels'].value_counts())
feature_df.to_csv('C:\\Users\\valer\\Desktop\\Thesis Software\\_bin\\feature_df.csv', index=False)

print('Execution terminated at: ', datetime.datetime.now())



