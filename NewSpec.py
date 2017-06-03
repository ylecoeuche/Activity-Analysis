"""
Author-Yannick Lecoeuche
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from astropy.io import fits as pf
from scipy import signal
import timeit
import shutil
from scipy import interpolate
from scipy import stats
from matplotlib import ticker
import pandas as pd

""" This purpose of this program is to take a collection of stellar observations
from the Lick Observatory and determine their relative activity values. These
values are compared to radial velocity and brightness measurements from other
observatories to determine possible correlation between the three stellar
measurements. """

def sort_obs(path):
    """ Takes a large folder of unsorted fits files and sorts them into their
    own folders by observatory, then takes a list of observations within a
    directory and creates subdirectories for individual stars. """

    if not os.path.exists('Keck Stars'):
        os.mkdir('Keck Stars')
    if not os.path.exists('Lick Stars'):
        os.mkdir('Lick Stars')
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    for f in files:
        if os.path.getsize(path+'/'+f) > 0:
            try:
                fits_file = pf.open(path + '/' + f)
                head = fits_file[0].header
                if "TARGNAME" in head:
                    os.rename(path+'/'+f, 'Keck Stars/'+f)
                else:
                    os.rename(path+'/'+f, 'Lick Stars/'+f)
            except:
                pass
    telescopes = ['Keck Stars', 'Lick Stars']
    for obs in telescopes:
        files = os.listdir(obs)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for f in files:
            if not os.path.isdir(obs+'/'+f):
                fits_file = pf.open(obs+'/'+f)
                head = fits_file[0].header
                if obs == telescopes[0]:
                    starname = head['TARGNAME']
                    if not os.path.exists(obs+'/'+starname):
                        os.mkdir(obs+'/'+starname)
                    os.rename(obs+'/'+f, obs+'/'+starname+'/'+f)
                if obs == telescopes[1]:
                    starname = head['OBJECT']
                    if not os.path.exists(obs+'/'+starname):
                        os.mkdir(obs+'/'+starname)
                    os.rename(obs+'/'+f, obs+'/'+starname+'/'+f)

def clean_up():
    """ Checks for files with only one observation and removes them.  """

    telescopes = ['Keck Stars', 'Lick Stars']
    for obs in telescopes:
        files = os.listdir(obs)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
            for f in files:
                items = os.listdir(obs+'/'+f)
                if '.DS_Store' in items:
                    items.remove('.DS_Store')
                if len(items) == 2:
                    shutil.rmtree(path+'/'+f)

def load_spectrum(path, order):
    """ Returns array of spectrum arrays at the specified order. """

    if os.path.isdir(path):
        fits_folder = os.listdir(path)
        spec_list = []
        if '.DS_Store' in fits_folder:
            fits_folder.remove('.DS_Store')
        for item in fits_folder:
            fits_file = pf.open(path + '/' + item)
            fits_spec = fits_file[0].data
            spec_list.append(fits_spec[order])

    else:
        fits_file = pf.open(path)
        fits_spec = fits_file[0].data
        spec_list = [fits_spec[order]]
    # for spec in spec_list:
    #     plt.plot(spec)
    # plt.show()

    return spec_list

def load_wave_sol(path, order):
    """ Return wavelength solution for either Lick or Keck observations at
    specified order. """

    if 'Lick Stars' in path:
        wav_sol_file = pf.open('dewar4_wls.fits')
        wave_sol_dat = wav_sol_file[0].data
        wave_sol = wave_sol_dat[order]

    if 'Keck Stars' in path:
        wav_sol_file = pf.open('keck_bwav.fits')
        wave_sol_dat = wav_sol_file[0].data
        wave_sol = wave_sol_dat[order]

    return wave_sol

def sig_noise(spec_list):
    """ Returns the signal to noise ratio as the square root of the median of the
    given spectra. """

    s_n = []
    for spec in spec_list:
        mdn = np.median(spec)
        s_n.append(np.sqrt(mdn))

    return s_n

def normalize(spec_list):
    """ Returns normalized spectra with single-pixel cosmic ray spikes removed.
    Normalization by average of highest 30 points (after cosmic ray removal). """

    for spec in spec_list:
        length = len(spec)
        for a in range(1, len(spec)-1):
            if spec[a] > (spec[a-1]+spec[a+1]):
                    spec[a] = (spec[a-1]+spec[a+1])*0.5

        if spec[0] > spec[1]*2:
            spec[0] = spec[1]

        if spec[length-1] > spec[length-2]*2:
            spec[length-1] = spec[length-2]

    norm_spec = []
    for spec in spec_list:
        arrange = np.sort(spec)
        avg = np.mean(arrange[-30:])
        norm_spec.append(spec/avg)
    # for p in norm_spec:
    #     plt.plot(p)
    # plt.show()

    return norm_spec

def rem_blaze(norm_spec, wave_sol, wav_ref, sun_spec, s_n):
    """ Removes blaze function by cross-correlating wave function to NSO Sun
    spectrum and shifting, dividing out the sun spectrum to remove spectral
    features, fitting a polynomial to the remaining curve, and dividng that
    from the original spectrum. Returns "flat" spectra and shifted wavelength
    solution. """

    flat_list = []
    pix_to_wav = wave_sol[1]-wave_sol[0]
    wav_min = min(wave_sol)
    wav_max = max(wave_sol)
    sun_min = np.argmin(abs(wav_ref - wav_min))
    sun_max = np.argmin(abs(wav_ref - wav_max))
    sun_ref = sun_spec[sun_min-100:sun_max+100]

    tck = interpolate.splrep(wav_ref[sun_min-100:sun_max+100], sun_ref)
    for l in range(len(norm_spec)):
        new_sun = interpolate.splev(wave_sol, tck)
        correlation = signal.correlate(new_sun, norm_spec[l], mode = 'full')
        # plt.plot(correlation)
        # plt.show()
        xcorr = np.arange(-len(norm_spec[l])+1, len(norm_spec[l]))
        x = np.argmax(correlation)
        n = 10
        centroid = np.sum(xcorr[x-n:x+n+1]*correlation[x-n:x+n+1])/np.sum(correlation[x-n:x+n+1])
        if abs(centroid) > 1:
            shift = int(centroid)
            new_sun = interpolate.splev(wave_sol+pix_to_wav*shift, tck) #HERE
        else:
            shift = 0
        if l == np.argmax(s_n) and s_n[l] > 20:
            true_wav = wave_sol+pix_to_wav*shift #HERE

        curve = norm_spec[l]/(new_sun)

        n = 3
        for i in range(n, len(curve)-n):
            curve[i] = np.mean(curve[i-n:i+n])

        xval = np.linspace(sun_min, sun_max, len(curve))
        coef = np.polyfit(xval, curve, 5)
        curve_fit = np.poly1d(coef)
        fit = curve_fit(xval)
        flat_list.append(norm_spec[l]/fit)
    #     plt.plot((wave_sol - pix_to_wav*centroid), norm_spec[l]/fit)
    # plt.show()

    try:
        true_wav
    except:
        true_wav = wave_sol

    return flat_list, true_wav

def correlate(flat_list, s_n, wave_sol):
    """ Cross correlates spectra within list to the one with the highest signal-
    to-noise ratio, if the maximum correlation is above a certain threshold.
    Returns the centroided shift for each spectrum. """

    n = 3
    smooth_list = []
    pix_to_wav = wave_sol[1]-wave_sol[0]
    ref = np.argmax(s_n)
    for spec in flat_list:
        smooth = np.array([])
        for i in range(n, (len(spec)-n)):
            smooth = np.append(smooth, np.mean(spec[i-n:i+n]))
        smooth_list.append(smooth)
        # plt.plot(smooth)
    # plt.show()

    offset = np.array([])
    cal_list = []
    for i in range(len(smooth_list)):
    # for i in range(len(flat_list)):
        correlation = signal.correlate(smooth_list[ref], smooth_list[i], mode = 'full')
        # correlation = signal.correlate(flat_list[ref], flat_list[i], mode = 'full')
        cal_list.append(correlation)
        if max(correlation) > 400:
            xcorr = np.arange(-len(flat_list[i])+1, len(flat_list[i]))
            x = np.argmax(correlation)
            n = 10
            # print (len(xcorr[x-n:x+n+1]))
            # print (len(correlation[x-n:x+n+1]))
            centroid = np.sum(xcorr[x-n:x+n+1]*correlation[x-n:x+n+1])/np.sum(correlation[x-n:x+n+1])
            cent_wav = centroid*pix_to_wav
            # offset = np.append(offset, cent_wav)
            offset = np.append(offset, cent_wav)
        else:
            offset = np.append(offset, 0)
    # print (offset)
    # for i in cal_list:
    #     plt.plot(i)
    # plt.show()
    return offset

def integration(path):
    """ Calls load_spectrum(), load_wave_sol(), sig_noise(), normalize(),
    rem_blaze(), and correlate() on fits files in orders 83-86 of path, then
    integrates the H and K spikes, and R and V bands to calculate S for the
    spectra. Returns array of S-values, average signal-to-noise ratio, and
    indices of spectra above s/n threshold. """

    sun = pf.open('nso.fits')
    wav_ref = sun[0].data
    sun_spec = sun[1].data
    items = os.listdir(path)

    if 'Lick Stars' in path:
        orders = [83, 84, 85, 86]

    if 'Keck Stars' in path:
        orders = [8, 7, 7, 6]

    H = []
    Hsn = []
    K = []
    Ksn = []
    R = []
    Rsn = []
    V = []
    Vsn = []
    s_ns = []
    okay = np.arange(0, len(items))
    bad = []

    for i in range(len(orders)):
        new_wave = []
        spec_list = load_spectrum(path, orders[i])
        wave_sol = load_wave_sol(path, orders[i])
        s_n = sig_noise(spec_list)
        if orders[i] == 83 or orders[i] == 8:
            Rsn.append(s_n)
        if orders[i] == 84 or orders[i] == 7:
            Hsn.append(s_n)
        if orders[i] == 85 or orders[i] == 7:
            Ksn.append(s_n)
        if orders[i] == 86 or orders[i] == 6:
            Vsn.append(s_n)
        for r in range(len(s_n)):
            if s_n[r] < 20:
                bad.append(r)
        norm_spec = normalize(spec_list)
        flat_list, true_wav = rem_blaze(norm_spec, wave_sol, wav_ref, sun_spec, s_n)
        offset = correlate(flat_list, s_n, wave_sol)
        for l in range(len(offset)):
            new_wave.append(offset[l] + true_wav) #HERE
        # for x in range(len(new_wave)):
        #     plt.plot(new_wave[x], flat_list[x])
        # plt.show()

        if orders[i] == 83 or orders[i] == 8:
            for b in range(len(flat_list)):
            # for b in range(len(norm_spec)):
                integ = []
                for j in range(len(new_wave[b])):
                # for j in range(len(wave_sol)):
                    if new_wave[b][j] >= 3991.07 and new_wave[b][j] <= 4011.07:
                    # if wave_sol[j] >= 3991.07 and wave_sol[j] <= 4011.07:
                        integ.append(flat_list[b][j])
                        # integ.append(norm_spec[b][j])
                R_val = np.sum(integ)
                R.append(R_val)

        if orders[i] == 84 or orders[i] == 7:
            for b in range(len(flat_list)):
            # for b in range(len(norm_spec)):
                integ = []
                for j in range(len(new_wave[b])):
                # for j in range(len(wave_sol)):
                    if new_wave[b][j] > 3968.5-2.18 and new_wave[b][j] < 3968.5:
                    # if wave_sol[j] > 3968.5-2.18 and wave_sol[j] < 3968.5:
                        weight = (.5/1.09)*new_wave[b][j]-1819.413
                        # weight = (.5/1.09)*wave_sol[j]-1819.413
                        integ.append(weight*flat_list[b][j])
                        # integ.append(weight*norm_spec[b][j])
                    if new_wave[b][j] < 3968.5+2.18 and new_wave[b][j] >= 3968.5:
                    # if wave_sol[j] < 3968.5+2.18 and wave_sol[j] >= 3968.5:
                        weight = (-.5/1.09)*new_wave[b][j]+1821.413
                        # weight = (-.5/1.09)*wave_sol[j]+1821.413
                        integ.append(weight*flat_list[b][j])
                        # integ.append(weight*norm_spec[b][j])
                H_val = np.sum(integ)
                H.append(H_val)

        if orders[i] == 85 or orders[i] == 7:
            for b in range(len(flat_list)):
            # for b in range(len(norm_spec)):
                integ = []
                for j in range(len(new_wave[b])):
                # for j in range(len(wave_sol)):
                    if new_wave[b][j] > 3933.7-2.18 and new_wave[b][j] < 3933.7:
                    # if wave_sol[j] > 3933.7-2.18 and wave_sol[j] < 3933.7:
                        weight = (.5/1.09)*new_wave[b][j]-1803.45
                        # weight = (.5/1.09)*wave_sol[j]-1803.45
                        integ.append(weight*flat_list[b][j])
                        # integ.append(weight*norm_spec[b][j])
                    if new_wave[b][j] < 3933.7+2.18 and new_wave[b][j] >= 3933.7:
                    # if wave_sol[j] < 3933.7+2.18 and wave_sol[j] >= 3933.7:
                        weight = (-.5/1.09)*new_wave[b][j]+1805.45
                        # weight = (-.5/1.09)*wave_sol[j]+1805.45
                        integ.append(weight*flat_list[b][j])
                        # integ.append(weight*norm_spec[b][j])
                K_val = np.sum(integ)
                K.append(K_val)

        if orders[i] == 86 or orders[i] == 6:
            for b in range(len(flat_list)):
            # for b in range(len(norm_spec)):
                integ = []
                for j in range(len(new_wave[b])):
                # for j in range(len(wave_sol)):
                    if new_wave[b][j] >= 3891.07 and new_wave[b][j] <= 3911.07:
                    # if wave_sol[j] >= 3891.07 and wave_sol[j] <= 3911.07:
                        integ.append(flat_list[b][j])
                        # integ.append(norm_spec[b][j])
                V_val = np.sum(integ)
                V.append(V_val)

    bad = list(set(bad))
    okay = np.delete(okay, bad)
    S = []
    for i in range(len(H)):
    # for i in range(len(norm_spec)):
        S_val = (H[i] + K[i])/(R[i]+V[i])
        S.append(S_val)
        sn_val = (Hsn[0][i] + Ksn[0][i] + Rsn[0][i] + Vsn[0][i])/4
        s_ns.append(sn_val)

    return S, s_ns, okay

def calibration():
    """ Calculates the calibration coefficient for S-values made by this program
    and values calculated by Keck observatory. Finds star overlap between Keck
    and Lick, separates observations taken within one day of one another, and
    fits a curve to the corresponding S-values. Plots the resulting calibration. """

    starnames, svals, dates = read_act()
    lick_names = os.listdir('Lick Stars')
    if '.DS_Store' in lick_names:
        lick_names.remove('.DS_Store')
    S_keck = []
    S_lick = []
    sn_tot = []
    success = 0
    error = 0
    starmatches = []
    for lname in lick_names:
        sn_all = []
        lick_star = []
        keck_star = []
        spec_match = []
        s_match = []
        for s in range(len(starnames)):
            if (lname.strip()).lower() == (starnames[s].strip()).lower():
                obs_names = os.listdir('Lick Stars/'+lname)
                if '.DS_Store' in obs_names:
                    obs_names.remove('.DS_Store')
                for h in range(len(obs_names)):
                    dat = pf.open('Lick Stars/'+lname+'/'+obs_names[h])
                    times = dat[0].header
                    time = times['DATE-OBS']
                    julian = to_julian(time)
                    # print (abs(julian - (dates[s]+2.44e6)))
                    if abs(julian - (dates[s]+2.44e6)) <= 0.5:
                        spec_match.append(h)
                        s_match.append(s)
                        pass

        if len(spec_match) > 0:
            try:
                S, s_ns, okay = integration('Lick Stars/'+lname)
                print (lname + " integrated correctly...")
                lick_temp = []
                sn_temp = []
                keck_temp = []
                for f in range(len(spec_match)):
                    for y in okay:
                        if y == spec_match[f]:
                            lick_temp.append(S[spec_match[f]])
                            keck_temp.append(svals[s_match[f]])
                            sn_temp.append(s_ns[spec_match[f]])
                if len(lick_temp) > 0:
                    S_lick.append(np.mean(lick_temp))
                    S_keck.append(np.mean(keck_temp))
                    sn_tot.append(np.mean(sn_temp))
                    success += 1
            except:
                error += 1
            # S, s_ns, okay = integration('Lick Stars/'+lname)
            # print (lname + " integrated correctly...")
            # lick_temp = []
            # sn_temp = []
            # keck_temp = []
            # for f in range(len(spec_match)):
            #     for y in okay:
            #         if y == spec_match[f]:
            #             S_lick.append(S[spec_match[f]])
            #             S_keck.append(svals[s_match[f]])
            #             sn_tot.append(s_ns[spec_match[f]])
            # success += 1


    print ('Successful Star Integrations: ' + str(success))
    print ('Failed Star Integrations: ' + str(error))

    x_min_y = []
    for i in range(len(S_keck)):
        x_min_y.append(S_keck[i]-S_lick[i])

    coef = np.polyfit(S_lick, S_keck, 1)
    xval = np.linspace(min(S_lick), max(S_lick), 100)
    curve_fit = np.poly1d(coef)
    print (curve_fit)

    chi2 = (stats.chisquare(S_lick, curve_fit(S_keck)))
    plt.plot(xval, curve_fit(xval), color = 'r')
    plt.text(0.30, 0.0280, 'y = ' + str(curve_fit))
    plt.text(0.30, 0.0275, 'Chi-squared fit: ' + str(chi2[0])[0:6])
    s = plt.scatter(S_lick, S_keck, c=sn_tot)
    # s = plt.scatter(S_keck, S_lick)
    cb = plt.colorbar(s)
    cb.set_label('Signal to Noise Ratio')
    plt.ylabel('Keck Activity')
    plt.xlabel('Lick Activity')
    plt.show()
    plt.plot(x_min_y, S_keck, 'ro')
    plt.xlabel('Keck - Lick Activity')
    plt.ylabel('Keck Activity')
    plt.show()

def s_n_histogram():
    """ Coadds single star spectra within the file "Lick Stars" and plots a
    histogram of spectra signal-to-noise ratios. """
    # Issue here is how to shift spectra of the same size and then coadd them.

    sun = pf.open('nso.fits')
    wav_ref = sun[0].data
    sun_spec = sun[1].data
    plt.plot([1,2,3])

    orders = [83, 84, 85, 86]
    bands = ['R', 'H', 'K', 'V']
    lick_names = os.listdir('Lick Stars')
    if '.DS_Store' in lick_names:
        lick_names.remove('.DS_Store')
    for i in range(len(orders)):
        s_ns = []
        for j in lick_names:
            path = 'Lick Stars/'+j
            spec_list = load_spectrum(path, orders[i])
            wave_sol = load_wave_sol(path, orders[i])
            for i in range(len(spec_list)):
                correlation = signal.correlate(spec_list[0], spec_list[i])

            red_sum = spec_list[0]
            for l in spec_list[1:]:
                red_sum += l
            red_sum /= len(spec_list)
            s_n = sig_noise([red_sum])
            s_ns.append(s_n[0])
        plt.subplot(2, 2, i+1)
        plt.hist(s_ns, bins = 20)
        plt.xlabel(bands[i] + ' SNR')
        plt.ylabel('Frequency')
    plt.show()

#def  coadd_std()

def Lick_to_Keck(xs):
    """ Converts Lick S-values to Keck S-values using the function calculated
    by calibration(). """

    ys = []
    for x in xs:
        y = 16.56*x - 0.1831
        ys.append(y)

    return ys

def to_julian(time):
    """ Given a Gregorian Calendar date string from the Lick fits file header,
    returns the corresponding Julian date. """

    import jdcal
    year = int(time[0:4])
    month = int(time[5:7])
    day = int(time[8:10])
    parts = jdcal.gcal2jd(year, month, day)
    julian = parts[0]+parts[1]-.5

    return julian

def read_act():
    """ Reads and returns columns from the Keck activity log.  """

    starnames = np.genfromtxt('Star Information/keck_activity_log.txt', dtype = str, skip_header = 1, usecols = 1)
    svals = np.genfromtxt('Star Information/keck_activity_log.txt', dtype = None, skip_header = 1, usecols = 2)
    dates = np.genfromtxt('Star Information/keck_activity_log.txt', dtype = None, skip_header = 1, usecols = 3)

    return starnames, svals, dates

def act_by_time(path):
    """ Shows the calibrated activity measurements of a given star as a function
     of time, using corresponding observations between Lick and Keck. """

    S, s_ns, okay = integration(path)
    starnames, svals, dates = read_act()

    s_keck = []
    date_keck = []
    times_list = []
    for s in range(len(starnames)):
        if (starnames[s].strip()).lower() in path.lower():
            s_keck.append(svals[s])
            date_keck.append(dates[s])
    obs_names = os.listdir(path)
    if '.DS_Store' in obs_names:
        obs_names.remove('.DS_Store')
    for h in range(len(obs_names)):
        dat = pf.open(path+'/'+obs_names[h])
        times = dat[0].header
        time = times['DATE-OBS']
        julian = to_julian(time)
        times_list.append(julian)
        # print (abs(julian - (dates[s]+2.44e6)))
    t_k = []
    t_l = []
    s_k = []
    s_l = []
    for j in range(len(date_keck)):
        for i in range(len(times_list)):
            if abs(times_list[i] - (date_keck[j]+2.44e6)) <= 0.25:
                t_k.append(date_keck[j]+2.44e6)
                s_k.append(s_keck[j])
                t_l.append(times_list[i])
                s_l.append(S[i]) #AVERAGE OUT THE MULTIPLE VALUES CLOSE TO A SINGLE KECK VALUE

    s_l = Lick_to_Keck(s_l)
    dat1, = plt.plot(t_k, s_k, 'bo')
    dat2, = plt.plot(t_l, s_l, 'ro')
    plt.legend([dat1, dat2], ['Keck', 'Lick'], loc=7)
    #
    # if '.DS_Store' in spec_list:
    #     spec_list.remove('.DS_Store')
    # times = []
    # for i in spec_list:
    #     dat = pf.open(path+'/'+i)
    #     info = dat[0].header
    #     time = info['DATE-OBS']
    #     julian = to_julian(time)
    #     times.append(julian)
    # plt.plot(times, S, 'bo')
    plt.xlabel('Julian Date [days]')
    plt.ylabel('S-value')
    plt.margins(0.05, 0.05)
    plt.show()  # FIX #FIX

def four_plot(path):
    """ Plots the normalized, flattened, and correlated R, V, H, and K continuum
    sections of the given stellar spectrum. """

    wav_start = [3980, 3948.5, 3913.7, 3880]
    wav_end = [4020, 3988.5, 3953.7, 3920]
    sec_start = [3991.07, 3966.5, 3931.7, 3891.07]
    sec_end = [4011.07, 3970.5, 3935.7, 3911.07]
    plt.plot([1,2,3])
    sun = pf.open('nso.fits')
    wav_ref = sun[0].data
    sun_spec = sun[1].data

    if 'Lick Stars' in path:
        orders = [83, 84, 85, 86]

    if 'Keck Stars' in path:
        orders = [8, 7, 7, 6]

    for i in range(len(orders)):
        new_wave = []
        spec_list = load_spectrum(path, orders[i])
        wave_sol = load_wave_sol(path, orders[i])
        if orders[i] == 83 or orders[i] == 8:
            s_n = sig_noise(spec_list, wave_sol)
        norm_spec = normalize(spec_list)
        flat_list, true_wav = rem_blaze1(norm_spec, wave_sol, wav_ref, sun_spec, s_n)
        offset = correlate(flat_list, s_n, wave_sol)
        for l in range(len(offset)):
            new_wave.append(offset[l] + true_wav)
        plt.subplot(2, 2, i+1)
        for j in range(len(flat_list)):
            plt.plot(new_wave[j], flat_list[j], 'k')
        plt.xlim(wav_start[i], wav_end[i])
        plt.ylim(0, 1.2)
        plt.axvline(sec_start[i], linestyle = '--')
        plt.axvline(sec_end[i], linestyle = '--')
        width = wav_end[i]-wav_start[i]
        plt.xticks([wav_start[i], wav_start[i]+width/4, wav_start[i]+2*width/4, wav_start[i]+3*width/4, wav_end[i]])
        plt.xlabel('Wavelength [Angstroms]')
        plt.ylabel('Brightness')
    plt.tight_layout()
    plt.show() # FIX

def act_overlay(path):
    """ Plots H and K activity values overlaid over one another. """

    wav_start = [3958.5, 3923.7]
    wav_end = [3978.5, 3943.7]
    plt.plot([1,2,3])

    if 'Lick Stars' in path:
        orders = [84, 85]

    if 'Keck Stars' in path:
        orders = [7, 7]

    for i in range(len(orders)):
        spec_list = load_spectrum(path, orders[i])
        wave_sol = load_wave_sol(path, orders[i])
        if orders[i] == 83 or orders[i] == 8:
            s_n = sig_noise(spec_list, wave_sol)
            for ratio in s_n:
                s_ns.append(ratio)
        norm_spec = normalize(spec_list)
        flat_list, true_wav = rem_blaze1(norm_spec, wave_sol, wav_ref, sun_spec, s_n)
        offset = correlate(flat_list, s_n, wave_sol)
        for l in range(len(offset)):
            new_wave.append(offset[l] + true_wav)
        plt.subplot(2, 1, i+1)
        for j in range(len(flat_list)):
            plt.plot(new_wave[j], flat_list[j], 'k', linewidth = 0.2)
        plt.xlim(wav_start[i], wav_end[i])
        plt.ylim(0, 1.0)
        width = wav_end[i]-wav_start[i]
        plt.xticks([wav_start[i], wav_start[i]+width/4, wav_start[i]+2*width/4, wav_start[i]+3*width/4, wav_end[i]])
        plt.xlabel('Wavelength [Angstroms]')
        plt.ylabel('Brightness')
    plt.tight_layout()
    plt.show() # FIX

def match():
    """ Attempts to remove blaze function by iterating through function
    parameters with the condition that the overlapping edges of the orders
    should be equal to one another. Doesn't work as well as other method. """

    left = load_spectrum('Keck Stars/3130-1591-1/bj93.1096.fits', 6)
    mid = load_spectrum('Keck Stars/3130-1591-1/bj93.1096.fits', 7)
    right = load_spectrum('Keck Stars/3130-1591-1/bj93.1096.fits', 8)

    orderl = 6
    orderm = 7
    orderr = 8

    l = normalize(left)[0]
    m = normalize(mid)[0]
    r = normalize(right)[0]

    lw = load_wave_sol('Keck Stars/3130-1591-1/bj93.1096.fits', 6)
    mw = load_wave_sol('Keck Stars/3130-1591-1/bj93.1096.fits', 7)
    rw = load_wave_sol('Keck Stars/3130-1591-1/bj93.1096.fits', 8)

    # plt.plot(lw, l, 'r', mw, m, 'b', rw, r, 'g')
    # plt.show()

    l_m = np.where(lw > mw[0])[0]
    m_l = np.where(mw < lw[-1])[0]
    m_r = np.where(mw > rw[0])[0]
    r_m = np.where(rw < mw[-1])[0]

    new_ml = m[:max(m_l)+1]
    new_r = r[:max(r_m)+1]

    tck1 = interpolate.splrep(lw[min(l_m):], l[min(l_m):])
    new_l = interpolate.splev(mw[:max(m_l)+1], tck1)
    tck2 = interpolate.splrep(mw[min(m_r):], m[min(m_r):])
    new_mr = interpolate.splev(rw[:max(r_m)+1], tck2)

    new_mwl = mw[:max(m_l)+1]
    new_lw = new_mwl
    new_rw = rw[:max(r_m)+1]
    new_mwr = new_rw

    # plt.plot(new_lw, new_l)
    # plt.plot(new_mwl, new_ml)
    # plt.show()

    cl = np.floor(len(lw)/2.0)
    cm = np.floor(len(mw)/2.0)
    cr = np.floor(len(rw)/2.0)
    centl = lw[int(cl)]
    centm = mw[int(cm)]
    centr = rw[int(cr)]

    Xl = orderl*(1-centl)/new_lw
    Xml = orderm*(1-centm)/new_mwl
    Xmr = orderm*(1-centm)/new_mwr
    Xr = orderr*(1-centr)/new_rw

    A = np.linspace(0.75, 1.0, 100)
    diffs = []
    for a in A:
        for b in A:
            for c in A:
                lfxn = (np.sin(a*np.pi*Xl))**2
                mlfxn = (np.sin(b*np.pi*Xml))**2
                mrfxn = (np.sin(b*np.pi*Xmr))**2
                rfxn = (np.sin(c*np.pi*Xr))**2
                l_flat = new_l/lfxn
                ml_flat = new_ml/mlfxn
                mr_flat = new_mr/mrfxn
                r_flat = new_r/rfxn
                diff = np.sum(abs(l_flat-ml_flat)) + np.sum(abs(mr_flat-r_flat))
                diffs.append(diff)
    best = str(np.argmin(diffs)).zfill(6)
    true_a = A[int(best[0:2])]
    true_b = A[int(best[2:4])]
    true_c = A[int(best[4:6])]
    print (true_a, true_b, true_c)
    Xl = orderl*(1-centl)/lw
    Xm = orderm*(1-centm)/mw
    Xr = orderr*(1-centr)/rw  #CORRECTLY COLLECT A B AND C
    lfxn = (np.sin(true_a*np.pi*Xl))**2 # MAKE THIS WORK FOR WHOLE ORDER NOW?
    mfxn = (np.sin(true_b*np.pi*Xm))**2
    rfxn = (np.sin(true_c*np.pi*Xr))**2
    plt.plot(lw, l/lfxn, 'r')
    plt.plot(mw, m/mfxn, 'g')
    plt.plot(rw, r/rfxn, 'b')
    plt.show()

def test():
    kobs = np.genfromtxt('Star Information/keck_log.txt', dtype = str, skip_header = 1, usecols = 0)
    knames = np.genfromtxt('Star Information/keck_log.txt', dtype = str, skip_header = 1, usecols = 1)
    kdates = np.genfromtxt('Star Information/keck_log.txt', dtype = None, skip_header = 1, usecols = 3)
    lobs = np.genfromtxt('Star Information/lick_log.txt', dtype = str, skip_header = 1, usecols = 0)
    lnames = np.genfromtxt('Star Information/lick_log.txt', dtype = str, skip_header = 1, usecols = 1)
    ldates = np.genfromtxt('Star Information/lick_log.txt', dtype = None, skip_header = 1, usecols = 3)
    Set = ['181253', '168874', '187237', '154345', '192020', '1326', '12051']
    kidx = []
    lidx = []
    for i in Set:
        for j in range(len(knames)):
            if i == knames[j].strip():
                kidx.append(j)
        for j in range(len(lnames)):
            if i == lnames[j].strip():
                lidx.append(j)

    kindx = []
    lindx = []
    for i in kidx:
        for j in lidx:
            if abs(kdates[i] - ldates[j]) < 1.5:
                kindx.append(i)
                lindx.append(j)

    col = []
    List = pd.DataFrame(0, index=range(len(lindx)), columns = col)
    List['Lick Observations'] = np.take(lobs, lindx)
    List['Lick Starnames'] = np.take(lnames, lindx)

    List.to_csv('Lick_Spectra_Request2.csv', index=False)
    # return np.take(lobs, lindx)
