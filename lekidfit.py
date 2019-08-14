#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:35:55 2019

@author: yvonne
Don't %run or similar this file unless you know what you're doing.
"""
import numpy as np
import scipy.special
#import scipy.integrate
import scipy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from lekidsim import lekid

#Utilities
def rotateIQ(I, Q):
    phi_rotate = calcPhase(I,Q)
    phi_rotate = np.median(phi_rotate)
    S21 = I + 1j*Q
    S21_rot = S21*np.exp(-1j*phi_rotate)
    I = np.real(S21_rot)
    Q = np.imag(S21_rot)
    return I, Q

def calcPhase(I, Q):
    phase = np.arctan2(Q, I)
    phase = correctRollover(phase, -np.pi/2., 0.9*np.pi, 2.*np.pi)
    return phase

def correctRollover(sig, low, high, ulim):
    mx = sig.max()
    mn = sig.min()
    if(mx>high and mn<low):
        w = np.where(sig < low)[0]
        sig[w] += ulim
    return sig

# fitting S21 curves
import lmfit

#resonance model
#following Eq 3.60 from Flanigan thesis
#note: needed to add linear slope fit to I and Q to fit resonances well.
def S21(freq, p):
    """
    Returns complex S21 values as calculated from lekid model.
    Input: array of frequencies freq (numpy array), array of parameters p (numpy array) in order Qr, Qc, C, A, normI, normQ, slopeI, slopeQ, interceptI, interceptQ
    Output: array of complex S21 values (numpy array)
    """
    Qr = p[0]# empirical
    Qc = p[1]# _should_ be constant
#    tau0 = p[1]
    tau0=2e-1
    C = p[2]
#    fr = p[2]# empirical
    A = p[3]# asymmetry factor
    normI = p[4]
    normQ = p[5]
    slopeI = p[6]
    slopeQ = p[7]
    interceptI = p[8]
    interceptQ = p[9]
    kid=lekid(tau0=tau0, C=C)
#    num = Qr/Qc * (1.+1.j*A)
#    den = 1.+2.j*Qr*(freq-fr)/fr
#    return norm*(1.-num/den) + slopeI*freq + 1.j*slopeQ*freq
    return (normI +1j*normQ)*(1 -(Qr*(1+1j*A))/(Qc*(1+2j*Qr*kid.fdet(freq)))) +(slopeI*(freq -freq[freq.shape[0]//2]) +interceptI +slopeI*freq[freq.shape[0]//2]) +1j*(slopeQ*(freq -freq[freq.shape[0]//2]) +interceptQ +slopeQ*freq[freq.shape[0]//2])# using fdet to approximate ffrac

#residual function for fitting
def resid(p, f, I, Q, stdI=1, stdQ=1):
    p = np.array([p['Qr'].value, p['Qc'].value, p['C'].value, p['A'].value, p['normI'].value, p['normQ'].value, p['slopeI'].value, p['slopeQ'].value, p['interceptI'].value, p['interceptQ'].value])
    s21 = S21(f,p)
#    s21mag = np.abs(s21)
#    s21phase = calcPhase(s21.real, s21.imag)
    res = np.append((s21.real -I)/stdI, (s21.imag -Q)/stdQ)
    return res

#fitting optical power model
a = lekid()
freq = np.linspace(5e8, 1e9, 10001)
popt=np.linspace(0,1e-11,11)
qr=np.zeros_like(popt)
tau0n=np.zeros_like(popt)
cn=np.zeros_like(popt)
aarray=np.zeros_like(popt)
norms=np.zeros_like(popt)
slopeIs=np.zeros_like(popt)
slopeQs=np.zeros_like(popt)
interceptIs=np.zeros_like(popt)
interceptQs=np.zeros_like(popt)

for i in range(popt.shape[0]):
    params = lmfit.Parameters()
    params.add('Qr', value=a.Qr(freq, popt[i]))
    params.add('Qc', value=2e-1)
    params.add('C', value=a.C)
    params.add('A', value=0.)
    params.add('normI', value=a.S21(freq, P_opt=popt[i]).real.max())
    params.add('normQ', value=a.S21(freq, P_opt=popt[i]).real.max())
    params.add('slopeI', value=0.)
    params.add('slopeQ', value=0.)
    params.add('interceptI', value=0.)
    params.add('interceptQ', value=0.)
    minner = lmfit.Minimizer(resid, params, fcn_args=(freq, a.S21(freq, P_opt=popt[i]).real, a.S21(freq, P_opt=popt[i]).imag))
    r = minner.minimize()
    p = np.array([r.params['Qr'].value,r.params['Qc'].value, r.params['C'].value,r.params['A'].value, r.params['normI'].value, r.params['normQ'].value, r.params['slopeI'].value, r.params['slopeQ'].value, r.params['interceptI'].value, r.params['interceptQ'].value])
    qr[i]=p[0]
    tau0n[i]=p[1]
    cn[i]=p[2]
    aarray[i]=p[3]
    norms[i]=p[4]
    slopeIs[i]=p[5]
    slopeQs[i]=p[6]
    interceptIs[i]=p[7]
    interceptQs[i]=p[8]

# fitting target sweep
import netCDF4
ncfile='toltec3_000440_00_0000_2019_05_22_20_16_46_targsweep.nc'
nc=netCDF4.Dataset(ncfile)
ufreq = np.unique(nc.variables['Data.Toltec.SweepFreq'][:].data)
Idat = nc.variables['Data.Toltec.Is'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Header.Toltec.ToneFreq'].shape[0])
Qdat = nc.variables['Data.Toltec.Qs'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Header.Toltec.ToneFreq'].shape[0])
#udat = nc.variables['Data.Toltec.Is'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Data.Toltec.Is'][:].data.shape[1]).mean(axis=1) +1j*nc.variables['Data.Toltec.Qs'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Data.Toltec.Qs'][:].data.shape[1]).mean(axis=1)
#ustd = nc.variables['Data.Toltec.Is'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Data.Toltec.Is'][:].data.shape[1]).std(axis=1) +1j*nc.variables['Data.Toltec.Qs'][:].data.reshape(ufreq.shape[0], -1, nc.variables['Data.Toltec.Qs'][:].data.shape[1]).std(axis=1)

a=lekid()
window=10
#notwindow=ufreq.shape[0]//2 -window
qr=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
#tau0n=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
qc=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
cn=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
aarray=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
normis=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
normqs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
slopeIs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
slopeQs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
interceptIs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
interceptQs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
rchis=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])

for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
#    print(i)
#    try:
    freq = nc.variables['Header.Toltec.ToneFreq'][i].data +ufreq
#        I,Q = rotateIQ(Idat[:,:,i], Qdat[:,:,i])
    I,Q = Idat[:,:,i], Qdat[:,:,i]
    uI,uQ = I.mean(axis=1), Q.mean(axis=1)
    stdI,stdQ = I.std(axis=1), Q.std(axis=1)
    minindex=np.where(np.abs(uI,uQ)[20:-20]==np.abs(uI,uQ)[20:-20].min())[0] +20
    windowlim=(max(minindex[0] -window,0), min(minindex[0] +window, freq.shape[0] -1))
    minfreq = freq[minindex]
    estc = 1/(4*np.pi**2*minfreq**2*(a.Lg +a.Lk()))
#        x=np.hypot(I,Q)
#        y=np.arctan2(Q,I)
    params = lmfit.Parameters()
    params.add('Qr', value=3e4)
    params.add('Qc', value=5e4)
    params.add('C', value=estc)
    params.add('A', value=0)
    params.add('normI', value=I.mean())
    params.add('normQ', value=Q.mean())
    params.add('slopeI', value=0)
    params.add('slopeQ', value=0)
    params.add('interceptI', value=0)
    params.add('interceptQ', value=0)
    minner = lmfit.Minimizer(resid, params, fcn_args=(freq, uI, uQ, stdI, stdQ))
    r = minner.minimize()
#        p = np.array([r.params['Qr'].value,r.params['tau0'].value, r.params['C'].value,r.params['A'].value, r.params['normI'].value, r.params['normQ'].value, r.params['slopeI'].value, r.params['slopeQ'].value, r.params['interceptI'].value, r.params['interceptQ'].value])
    qr[i] = r.params['Qr'].value#p[0]
    qc[i] = r.params['Qc'].value#p[1]
    cn[i] = r.params['C'].value#p[2]
    aarray[i] = r.params['A'].value#p[3]
    normis[i] = r.params['normI'].value#p[4]
    normqs[i] = r.params['normQ'].value#p[5]
    slopeIs[i] = r.params['slopeI'].value#p[6]
    slopeQs[i] = r.params['slopeQ'].value#p[7]
    interceptIs[i] = r.params['interceptI'].value#p[8]
    interceptQs[i] = r.params['interceptQ'].value#p[9]
    rchis[i] = r.redchi
#    except:
#        pass

for i in range(nc.variables['Data.Toltec.SweepFreq'].shape[0]):
#    if qr[i]!=0:
    freq = nc.variables['Header.Toltec.ToneFreq'][i].data +ufreq
#    I,Q = rotateIQ(Idat[:,:,i], Qdat[:,:,i])
    I,Q = Idat[:,:,i], Qdat[:,:,i]
    uI,uQ = I.mean(axis=1), Q.mean(axis=1)
    stdI,stdQ = I.std(axis=1), Q.std(axis=1)
    model = S21(freq, [qr[i], qc[i], cn[i], aarray[i], normis[i], normqs[i], slopeIs[i], slopeQs[i], interceptIs[i], interceptQs[i]])
    
    fig,ax1 = plt.subplots()
    ax1.errorbar(freq, uI, yerr=stdI, ls='', marker='x', capsize=2,  color='blue', label='I')
    ax1.plot(freq, model.real, color='blue', label='S21(real)')
    ax1.set_xlabel('freq/Hz')
    ax1.set_ylabel('I')
    ax1.set_title('i={}, redchi={}'.format(i, rchis[i]))
    ax2 = ax1.twinx()
    ax2.errorbar(freq, uQ, yerr=stdQ, ls='', marker='x', capsize=2,  color='orange', label='Q')
    ax2.plot(freq, model.imag, color='orange', label='S21(imag)')
    ax2.set_ylabel('Q')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.savefig('lekidfit_obs440_{0:d}.png'.format(i), bbox_inches='tight')
    plt.close()

def S21e(freq, p):
    """
    Returns complex S21 values from Zhiyuan's fits.
    Input: array of frequencies freq (numpy array), array of parameters p (numpy array) in order Qr, Qc, fr, A, normI, normQ, slopeI, slopeQ, interceptI, interceptQ
    Output: array of complex S21 values (numpy array)
    """
    Qr = p[0]# empirical
    Qc = p[1]# _should_ be constant
#    tau0 = p[1]
#    C = p[2]
    fr = p[2]# empirical
    A = p[3]# asymmetry factor
    normI = p[4]
    normQ = p[5]
    slopeI = p[6]
    slopeQ = p[7]
    interceptI = p[8]
    interceptQ = p[9]
#    kid=lekid(tau0=tau0, C=C)
    num = Qr/Qc * (1.+1.j*A)
    den = 1.+2.j*Qr*(freq-fr)/fr
    return (normI +1j*normQ)*(1.-num/den) +(slopeI*(freq -freq[freq.shape[0]//2]) +interceptI +slopeI*freq[freq.shape[0]//2]) +1j*(slopeQ*(freq -freq[freq.shape[0]//2]) +interceptQ +slopeQ*freq[freq.shape[0]//2])# using fdet to approximate ffrac
#    return (normI +1j*normQ)*(1 -(Qr*(1+1j*A))/(kid.Qc*(1+2j*Qr*kid.fdet(freq)))) +(slopeI*(freq -freq[freq.shape[0]//2]) +interceptI +slopeI*freq[freq.shape[0]//2]) +1j*(slopeQ*(freq -freq[freq.shape[0]//2]) +interceptQ +slopeQ*freq[freq.shape[0]//2])# using fdet to approximate ffrac

# comparing with zhiyuan's fits
zfits=np.genfromtxt('toltec3_000440_00_0000_2019_05_22_20_16_46_targsweep.txt')
a=lekid()
#window=10
#notwindow=ufreq.shape[0]//2 -window
qr=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
#tau0n=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
qc=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
cn=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
aarray=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
normis=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
normqs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
slopeIs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
slopeQs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
interceptIs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
interceptQs=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])
rchis=np.zeros(nc.variables['Header.Toltec.ToneFreq'].shape[0])

for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
#    print(i)
#    try:
    freq = nc.variables['Header.Toltec.ToneFreq'][i].data +ufreq
#        I,Q = rotateIQ(Idat[:,:,i], Qdat[:,:,i])
    I,Q = Idat[:,:,i], Qdat[:,:,i]
    uI,uQ = I.mean(axis=1), Q.mean(axis=1)
    stdI,stdQ = I.std(axis=1), Q.std(axis=1)
#    minindex=np.where(np.abs(uI,uQ)[20:-20]==np.abs(uI,uQ)[20:-20].min())[0] +20
#    windowlim=(max(minindex[0] -window,0), min(minindex[0] +window, freq.shape[0] -1))
#    minfreq = freq[minindex]
    estc = 1/(4*np.pi**2*zfits[i,10]**2*(a.Lg +a.Lk()))
#        x=np.hypot(I,Q)
#        y=np.arctan2(Q,I)
    params = lmfit.Parameters()
    params.add('Qr', value=zfits[i,8])
    params.add('Qc', value=5e4)
    params.add('C', value=estc)
    params.add('A', value=zfits[i,11])
    params.add('normI', value=zfits[i,12])
    params.add('normQ', value=zfits[i,13])
    params.add('slopeI', value=zfits[i,14])
    params.add('slopeQ', value=zfits[i,15])
    params.add('interceptI', value=zfits[i,16])
    params.add('interceptQ', value=zfits[i,17])
    minner = lmfit.Minimizer(resid, params, fcn_args=(freq, uI, uQ, stdI, stdQ))
    r = minner.minimize()
#        p = np.array([r.params['Qr'].value,r.params['tau0'].value, r.params['C'].value,r.params['A'].value, r.params['normI'].value, r.params['normQ'].value, r.params['slopeI'].value, r.params['slopeQ'].value, r.params['interceptI'].value, r.params['interceptQ'].value])
    qr[i] = r.params['Qr'].value#p[0]
    qc[i] = r.params['Qc'].value#p[1]
    cn[i] = r.params['C'].value#p[2]
    aarray[i] = r.params['A'].value#p[3]
    normis[i] = r.params['normI'].value#p[4]
    normqs[i] = r.params['normQ'].value#p[5]
    slopeIs[i] = r.params['slopeI'].value#p[6]
    slopeQs[i] = r.params['slopeQ'].value#p[7]
    interceptIs[i] = r.params['interceptI'].value#p[8]
    interceptQs[i] = r.params['interceptQ'].value#p[9]
    rchis[i] = r.redchi
    
for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
#    if qr[i]!=0:
    freq = nc.variables['Header.Toltec.ToneFreq'][i].data +ufreq
#    I,Q = rotateIQ(Idat[:,:,i], Qdat[:,:,i])
    I,Q = Idat[:,:,i], Qdat[:,:,i]
    uI,uQ = I.mean(axis=1), Q.mean(axis=1)
    stdI,stdQ = I.std(axis=1), Q.std(axis=1)
    model = S21(freq, [qr[i], qc[i], cn[i], aarray[i], normis[i], normqs[i], slopeIs[i], slopeQs[i], interceptIs[i], interceptQs[i]])
#    model = S21e(freq, zfits[i,8:])
#    rchi = np.linalg.norm(((uI +1j*uQ) -model)**2/(stdI +1j*stdQ)**2)/(freq.shape[0] -10)
    
    fig,ax1 = plt.subplots()
    ax1.errorbar(freq, uI, yerr=stdI, ls='', marker='x', capsize=2,  color='blue', label='I')
    ax1.plot(freq, model.real, color='blue', label='S21(real)')
    ax1.set_xlabel('freq/Hz')
    ax1.set_ylabel('I')
    ax1.set_title('i={}, redchi={}'.format(i, rchis[i]))
    ax2 = ax1.twinx()
    ax2.errorbar(freq, uQ, yerr=stdQ, ls='', marker='x', capsize=2,  color='orange', label='Q')
    ax2.plot(freq, model.imag, color='orange', label='S21(imag)')
    ax2.set_ylabel('Q')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.savefig('lekidfit_obs440_{0:d}.png'.format(i), bbox_inches='tight')
    plt.close()



# plotting np.abs(s21)**2 and np.abs(d21)
# not in dB
for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
    plt.plot(ufreq+nc.variables['Header.Toltec.ToneFreq'][:].data[i], np.abs(Idat[:,:,i].mean(axis=1),Qdat[:,:,i].mean(axis=1))**2)

for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
    plt.plot(ufreq+nc.variables['Header.Toltec.ToneFreq'][:].data[i], np.abs(np.gradient(Idat[:,:,i].mean(axis=1) +1j*Qdat[:,:,i].mean(axis=1))))

# coadding d21
freqs=np.linspace(4.5e8,9e8,5001)
interps=[]
flims=[]
d21=[]

for i in range(nc.variables['Header.Toltec.ToneFreq'].shape[0]):
    fs = ufreq +nc.variables['Header.Toltec.ToneFreq'][:].data[i]
    flims.appaned((fs[0],fs[-1]))
    interps.append(interp1d(fs, np.gradient(Idat[:,:,i].mean(axis=1) +1j*Qdat[:,:,i].mean(axis=1))))

for f in freqs:
    z=[]
    for i in range(len(flims)):
        if f>=flims[i][0] and f<flims[i][-1]:
            interpfunc=interps[i]
            z.append(interpfunc(f))
        else:
            pass
    d21.append(np.mean(z))


"""
# fitting using lmfit.Model
def S21m(freq, Qr, tau0, C, A, normI, normQ, slopeI, slopeQ, interceptI, interceptQ):
    kid=lekid(tau0=tau0, C=C)
#    num = Qr/Qc * (1.+1.j*A)
#    den = 1.+2.j*Qr*(freq-fr)/fr
#    return norm*(1.-num/den) + slopeI*freq + 1.j*slopeQ*freq
    return (normI +1j*normQ)*(1 -(Qr*(1+1j*A))/(kid.Qc*(1+2j*Qr*kid.fdet(freq)))) +(slopeI*freq +interceptI) +1j*(slopeQ*freq +interceptQ)# using fdet to approximate ffrac

s21model=lmfit.Model(S21m)
s21model.set_param_hint('tau0', value=2e-1, vary=False)

for i in range(np.unique(nc.variables['Data.Toltec.SweepFreq'][:].data).shape[0]):
    print(i)
    freq = nc.variables['Header.Toltec.ToneFreq'][i].data +ufreq
    I,Q = rotateIQ(Idat[:,:,i], Qdat[:,:,i])
    uI,uQ = I.mean(axis=1), Q.mean(axis=1)
    stdI,stdQ = I.std(axis=1), Q.std(axis=1)
    params = s21model.make_params(Qr=3e4, C=5e-12, A=0)
"""
    