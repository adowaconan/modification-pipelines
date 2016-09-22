# modification-pipelines
## progress of how the pipeline changes and how logistic regression can be used to detect spindles in EEG channel data

Pipelines for analyzing EEG data and detect spindls, sleeping stages, and k-complex. 

## Entering the phase using threshold detection algorithm to detect spindles in 6 EEG channels. 
 1. Threshold detection algorithm
 2. ASI capturing alpha,beta,spindle activity in terms of power density


# [ICA parameters](martinos.org/mne/stable/auto_tutorials/plot_artifacts_correlation_ica.html):
1. bandpass 1-200 Hz
2. apply notch filter at 60 Hz
3. MNE ICA: iterration = 3000, fixed random state
4. artifact dectection is based on channles "LOC" and "ROC", and run automatically
5. rejection parameters: EEG: 180 - 360 depending on subjects; tstep: 2 seconds; EOG criteria: 0.4; skewness: 2; kurt: 2; variance:2
6. bandpass 1-50 Hz

# Detecting spindles:
1. bandpass slow/fast spindle range (10-12Hz/12.5-14.5Hz) [(Begmann et al., 2012)](http://www.ncbi.nlm.nih.gov/pubmed/22037418])
2. select channels: F3, F4, C3, C4, O1, O2
3. use a moving window to compute root-mean-square (RMS): Gaussian window, standard deviation = windown length / .68 / 2, window length = 200 samples, convolution using central part of convolution of the same size
4. compute the harmonic mean of the RMSs of the 6 channels and call it the mean channel
5. compute RMS for the mean channel
6. compute trimmed mean and trimmed standard deviation (5%) on the data after the first 100 seconds and before the last 30 seconds for both the individual channels and the mean channel
7. threshold = mean + .9 * standard deviation (Begmann et al., 2012)
8. post threshold parameter: segments that is above the threshold and duration of the segments is in between 0.5 - 2 secs
9. determining spindles: find spindles in AT LEAST (>=) 3 channels AND find spindle in average channel at the similar time stamp (deviate < 1 second)

# [Power spetral density analysis](spisop.org/documentation)
1. delta 1: 0-2 Hz
2. delta 2: 2-4 Hz
3. theta: 4-8 Hz
4. alpha: 8-12 Hz
5. beta: 12-20 Hz
6. slow spindle: 10-12 Hz
7. fast spindle: 12.5-14.5 Hz

compute power spectral density of all these frequency bands using a moving window with lenght of 10 seconds and overlapping half of the window size.
power spectral density is rescaled by 10*log10

# Assumptions:
1. alpha drop == awake to stage 1 sleep
2. alpha drop and beta drop == stage 1 to stage 2 sleep
3. alpha drop and beta drop, and slow/fast spindle increase == generating spindles

# To capture my assumption: compute ASI
