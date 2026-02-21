import numpy as np
from scipy import interpolate

def transitions_times(t, y, threshold):
    # Find where y crosses the threshold (increasing).
    lower = y < threshold
    higher = y >= threshold
    transition_indices = np.where(lower[:-1] & higher[1:])[0]
    transition_times=t[transition_indices]

    return transition_times

def calc_meanfreq(t_periods,n_pulse):
    ## Calculation of the mean frequency obtained by the shaft encoder signal
    deltas=np.diff(t_periods[::n_pulse])
    frequencies=1/deltas
    freq_mean = np.mean(frequencies)

    return freq_mean

def calc_deltatheta(mean_freq,fs,sampling_method):
    ## Calculation of the increment in angle domain
    if sampling_method=='mean_freq':
        delta_theta=((2*np.pi)/(int(fs/mean_freq)))
    elif sampling_method=='temporal_fs':
        delta_theta=(2*np.pi)/(fs)

    return delta_theta

def timephase_map(delta_theta,t_periods, n_pulse):
    ## Calculation of the time instanst related to the regular increments of angle
    theta=np.linspace(0,((2*np.pi/n_pulse)*(len(t_periods)-1)),num=len(t_periods))
    f_theta=interpolate.interp1d(theta,t_periods,'cubic')
    theta_inc=np.linspace(0,((2*np.pi/n_pulse)*(len(t_periods)-1)),num=(round(((2*np.pi)/(n_pulse))/(delta_theta))*(len(t_periods)-1))+1)
    times_inc=f_theta(theta_inc)

    return times_inc

def angular_resample(times_inc,signal_time,time):
    ## Resampling the signal in angle domain with regular increments
    f=interpolate.interp1d(time,signal_time,'cubic')
    signal_ang=f(times_inc)   

    return signal_ang