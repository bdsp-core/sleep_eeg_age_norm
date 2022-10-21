import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from scipy.signal import find_peaks


def find_spectral_peak(spec_db_, freq_, lb=5.5, ub=12):
    ids = (freq_>=lb)&(freq_<=ub)
    spec_db = spec_db_[ids]
    freq = freq_[ids]
    
    dfreq = np.median(np.diff(freq))
    func = UnivariateSpline(freq, spec_db, k=3, s=0.1)
    spec_db2 = func(freq)
    res = linregress(freq, spec_db2)
    spec_db_diff = spec_db2 - (freq*res.slope+res.intercept)
    peaks, _ = find_peaks(spec_db_diff, width=int(round(1/dfreq)))
    if len(peaks)==1:
        if peaks[0]==0 or peaks[0]==len(spec_db_diff)-1:
            iaf = np.nan
        else:
            iaf = freq[peaks[0]]
    else:
        iaf = np.nan
        
    if not np.isnan(iaf):
        ids = (freq>=iaf-1)&(freq<=iaf+1)
        spec_db3 = spec_db2[ids]
        if np.mean(np.diff(spec_db3)>=0)<0.9 and np.mean(np.diff(spec_db3)<=0)<0.9:
            iaf = freq[ids][np.argmax(spec_db3)]
            
    return iaf
    
