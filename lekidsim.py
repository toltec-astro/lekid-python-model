#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 02:25:15 2018

@author: yvonne
"""
import numpy as np
import scipy.special
#import scipy.integrate
import scipy.constants as const
#import matplotlib.pyplot as plt

#TODO: initialise many at once? need detector array class? or just use list
class lekid(object):
    """#TODO: better docstring
    NOTE: Steady-state only. Does not describe time-dependent perturbations.
    current critical values are for TiN/Ti/TiN trilayers unless otherwise labelled. 
    """
    def __init__(self,
                 wI = 4e-6, # metre: Inductor width
                 lI = 4e-4, # metre: Inductor length
                 tI = 1.8e-8, # metre: Inductor thickness
#                 lambL = 5e-8, # metre: penetration depth of Al
                 Tc = 1.4, # Kelvin: critical temperature
                 tau0 = 1e-9, #4.38e-7, # second: characteristic electron-phonon interaction time of Al
                 T = 0.1, # Kelvin: Temperature of the LeKID
                 R_nsq = 9e1,#1.7536e4, # Ohms per square: normal sheet resistivity just above Tc
                 lamb = 1.1e-3, # metre: optical wavelength
                 delta_nuopt = 1.0e10, #4e11, Optical Bandwidth (Hertz) #thereabouts
                 Lg = 3e-9, # henry: geometric inductance #TODO: derivable?
                 C = 5e-12, # farads: capacitance
                 eta_opt = 0.8, # optical efficiency   
                 eta_pb = 0.57, # pair-breaking efficiency
                 tau_phon_br = 1e-10, # seconds: time for phonon of sufficient energy to break Cooper pair of Al
                 Qc = 5e4, # coupling quality factor
                 Ql = 2e4, # loss quality factor
#                 Lksq = 9e-11, # sheet kinetic inductance per square (Henry/square)
#                 nqp0 = 1e20,# m^-3: baseline unloaded quasiparticle density of TiN
                 N0 = 8.277e+46,# per Joule per metre cubed: single-spin density of electron states of TiN at Fermi energy
#                 Nsq = 5333.33, # Number of squares 
#                 Lk_tot = 9.86e-9, # Total Kinetic Inductance (Henry)
                 f_read = 1e9, # Hertz: READOUT frequency of circuit
                 Z_line = 50.0, # Transmission line impedence (Ohms)
#                 alpha = 0.123, # effective kinetic inductance fraction (?)
                 P_read = 6.31e-12, # Readout Power (Watts)
                 T_amp = 3.0, # Amplifier Noise Temperature (Kelvin)
                 deltav_read = 50.0, # Readout Bandwidth (Hertz)
                 V_read = 1.776e-5, # readout voltage (volts)
                 ):
        self.wI = wI
        self.lI = lI
        self.tI = tI
#        self.VI = self.wI*self.lI*self.tI# metre^3: sheet volume
        self.Tc = Tc
        self.tau0 = tau0
        self.T = T
        self.R_nsq = R_nsq
        self.lamb = lamb
#        self.nu_opt = const.c/self.lamb# Hertz: Optical frequency
        self.delta_nuopt = delta_nuopt
#        self.E_gamma = const.h*self.nu_opt# energy of photon
#        self.delta0 = 1.764*const.k*Tc# Joule: gap energy at 0K
#        self.delta = self.delta0*(1-np.sqrt(2*const.pi*const.k*self.delta0*self.T)*np.exp(-self.delta0/(const.k*self.T)))# Joule: gap energy at T (identical to delta0)#NOTE: I've been using delta0 and delta interchangeably
#        self.sigma_n = sigma_nsq/self.tI# siemens per metre: normal conductivity just above Tc (5e9)
#        self.lambL = const.hbar/(const.pi*const.mu_0*self.tI*self.delta*self.sigma_n)#lambL# metre: penetration depth
#        self.nqp0 = nqp0# from Jay: baseline unloaded QP density, assumed from literature
        self.N0 = N0# per Joule per metre cubed: single-spin density of electron states at Fermi energy
        self.Lg = Lg
        self.C = C
        self.eta_opt = eta_opt
        self.eta_pb = eta_pb
#        self.N_qp_photon = self.eta_pb*const.h*self.nu_opt/self.delta#N_qp_photon# Number of quasi particles produced per photon
        self.tau_phon_br = tau_phon_br
        self.Qc = Qc
        self.Ql = Ql
#        self.Lksq = Lksq
        
#        self.Lk_tot = Lk_tot
#        self.alpha = alpha
        self.f_read = f_read
        self.Z_line = Z_line
        self.P_read = P_read
        self.T_amp = T_amp
        self.deltav_read = deltav_read
        self.V_read = V_read
#        self.P_opt = P_opt#why do functions require an external P_opt entry
        
    # constants
    @property
    def VI(self):
        """
        Return the volume of the inductor.
        Inputs: none
        """
        return self.wI*self.lI*self.tI# metre^3
    
    @property
    def nsq(self):
        """
        Return the number of squares of the inductor.
        Inputs: none
        """
        return self.lI/self.wI
    
    @property
    def nu_opt(self):
        """
        Return the optical frequency of detected light.
        Inputs: none
        """
        return const.c/self.lamb# hertz
    
    @property
    def E_gamma(self):
        """
        Return the energy of photon.
        Inputs: none
        """
        return const.h*self.nu_opt# joule
    
#    @property
#    def fread(self):
#        """
#        Return the default readout frequency of circuit, where fnew(P_opt=0) should be.
#        Inputs: none
#        """
#        Lk_0=1/(4*np.pi**2*self.C*2.5e17) -self.Lg
#        return 1/(2*np.pi*np.sqrt(self.C*(Lk_0+self.Lg)))# hertz
    
    @property
    def delta0(self):
        """
        Return the gap energy at 0K.
        Inputs: none
        """
        return 1.764*const.k*self.Tc# Joule
    
    @property
    def delta(self):#I've been using delta0 and delta interchangeably
        """
        Return the gap energy at T. (almost identical to delta0)
        Inputs: none
        """
        return self.delta0*(1 -np.sqrt(2*const.pi*const.k*self.delta0*self.T)*np.exp(-self.delta0/(const.k*self.T)))# Joule
    
    @property
    def sigma_n(self):
        """
        Return the normal conductivity just above Tc (5e9).
        Inputs: none
        """
        return 1/(self.R_nsq*self.tI)# siemens per metre
    
    @property
    def lambL(self):
        """
        Return the penetration depth in the thin film limit.
        Inputs: none
        """
        return const.hbar/(const.pi*const.mu_0*self.tI*self.delta*self.sigma_n)# metre
    
#    @property
#    def N0(self):
#        """
#        Return the single-spin density of electron states at Fermi energy. (from n_qp_ss=1e20, setting F_phon=1)
#        Inputs: none
#        """
#        return self.nqp0/(2*np.sqrt(2*np.pi*const.k*self.T*self.delta0)*np.exp(-self.delta0/(const.k*self.T)))# per joule per metre^3
    
    @property
    def N_qp_photon(self):
        """
        Return the number of quasiparticles produced per photon.
        Inputs: none
        """
        return self.eta_pb*const.h*self.nu_opt/self.delta
    
    # intrinsic: depend on properties of material
    @property
    def R_qp(self):
        """
        Return the intrinsic quasiparticle recombination constant.
        Inputs: none
        """
        return 2*(self.delta0/(const.k*self.Tc))**3/(self.N0*self.delta0*self.tau0)# units?
    
    # thermal: depends on self.T
    @property
    def n_qp_therm(self):# TODO: change approximation to K?
        """
        Return the density of quasiparticles arising due to thermal effects at temperature of resonator. Differs from n_qp_ss by phonon trapping factor F_phon.
        Inputs: none
        """
        return 2*self.N0*np.sqrt(2*np.pi*const.k*self.T*self.delta0)*np.exp(-self.delta0/(const.k*self.T))# metre^-3
    
    @property
    def gamma_G(self):# TODO: change approximation to K?
        """
        Return the low-temperature thermal generation rate at temperature of resonator.
        Inputs: none
        """
        return 16*self.N0*self.delta0**3*np.pi*self.T*np.exp(-2*self.delta0/(const.k*self.T))/(self.tau0*const.k**2*self.Tc**3)# per second(?)
    
    # steady-state: in terms of time
    @property
    def tau_phon_es(self):
        """
        Return the phonon escape time [in seconds]. (redundant)
        Inputs: none
        """
        eta_phon_trans = 1e-9# transmission probability per encounter #TODO: find a value for this
        s = 6.4e3# metre per second (speed of sound in Al, most like)
        return 4*self.tI/(eta_phon_trans*s)# second
    
    @property
    def F_phon(self):
        """
        Return the phonon trapping factor. (redundant)
        Inputs: none
        """
        return 1 +self.tau_phon_es/self.tau_phon_br# units?
    
    @property
    def R_eff(self):
        """
        Return the effective quasiparticle recombination constant.
        Inputs: none
        """
        return self.R_qp#/self.F_phon# units?
    
    @property
    def tau_qp(self):
        """
        Return the quasiparticle relaxation time. (redundant)
        Inputs: none
        """
        return (4*self.R_eff*self.gamma_G)**(-0.5)# second
    
    @property
    def n_qp_ss(self):
        """
        Return the steady-state quasiparticle density.
        Inputs: none
        """
        return np.sqrt(self.gamma_G/self.R_eff)# metre^-3
        #return self.nqp0# metre^-3
    
    @property
    def N_qp_ss(self):
        """
        Return the total NUMBER of quasiparticles in resonator in steady-state.
        Inputs: none
        """
        return self.VI*np.sqrt(self.gamma_G/self.R_eff)

    # optical: depends on optical power
    def Gamma_opt(self, P_opt=0):
        """
        Return the background optical power quasiparticle generation rate. CURRENTLY ASSUMES OPTICAL EFFICIENCY ACCOUNTED FOR.#TODO: FIX FOR ACTUAL MODELLING
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return P_opt*self.eta_pb/self.delta#*self.dPabs_dPinc 
    
    # all together now
    def N_qp_tot(self, P_opt=0):
        """
        Return the total NUMBER of quasiparticles in resonator due to thermal and BACKGROUND optical power effects.
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return np.sqrt((self.VI**2*self.gamma_G +self.VI*self.Gamma_opt(P_opt=P_opt))/self.R_eff)# metre^-3
    
    # conductivity
    def sigma1_0(self, f):
        """
        Return the real part of complex conductivity at T=0K.
        Inputs: f (Hz: frequency)
        """
        return 0# ohms?
    
    def sigma2_0(self, f):
        """
        Return the imag part of complex conductivity at T=0K.
        Inputs: f (Hz: frequency)
        """
        return np.pi*self.delta0*self.sigma_n/(f*const.h)# ohms?

    def sigma1rat(self, f):
        """
        Return the ratio of the real part of complex conductivity to quasiparticle density response at temperature of resonator. (used in responsivity)
        Inputs: f (Hz: frequency)
        """
        return (8*self.delta0/(np.pi**3*const.k*self.T))**(0.5)*np.sinh(const.h*f/(2*const.k*self.T))*scipy.special.kv(0, const.h*f/(2*const.k*self.T))
    
    def sigma2rat(self, f):
        """
        Return the ratio of the imag part of complex conductivity to quasiparticle density response at temperature of resonator. (used in responsivity)
        Inputs: f (Hz: frequency)
        """
        return -1 -(2*self.delta0/(np.pi*const.k*self.T))**(0.5)*np.exp(-const.h*f/(2*const.k*self.T))*scipy.special.iv(0, const.h*f/(2*const.k*self.T))

    def sigma1(self, f, P_opt=0):
        """
        Return the real part of complex conductivity at temperature of resonator.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        #return self.sigma1rat()*self.N_qp_tot(P_opt=P_opt) + self.sigma1_0()# ohms?
        return self.N_qp_tot(P_opt=P_opt)*self.dsig1_dN(f) +self.sigma1_0(f)# ohms?
    
    def sigma2(self, f, P_opt=0):
        """
        Return the imag part of complex conductivity at temperature of resonator.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        #return self.sigma2rat()*self.N_qp_tot(P_opt=P_opt) + self.sigma2_0()# ohms?
        return self.N_qp_tot(P_opt=P_opt)*self.dsig2_dN(f) +self.sigma2_0(f)# ohms?
    
    def sigma(self, f, P_opt=0):
        """
        Return the complex conductivity at temperature of resonator.
        Inputs:  f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.sigma1(f, P_opt=P_opt) -1j*self.sigma2(f, P_opt=P_opt)

    # impedance
    def Zs_0(self, f):
        """
        Return the surface impedance in thin film local limit at T=0K.
        Inputs: f (Hz: frequency)
        """
        #return 1j*self.Xs_0(f)
        return 1j/(self.tI*self.sigma2_0(f))# ohms?/sq
    
    def Zs(self, f, P_opt=0):
        """
        Return the surface impedance in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        #return self.sigma1(P_opt=P_opt)/(self.tI*1j*self.sigma2(P_opt=P_opt)**2) + 1j/(self.tI*1j*self.sigma2(P_opt=P_opt))# units?
        return 1/(self.tI*self.sigma(f, P_opt=P_opt))# ohms?/sq
        #return self.Zs_0(f)*(1 +(self.sigma(f, P_opt=P_opt) +1j*self.sigma2_0(f))/(1j*self.sigma2_0(f)))# units?

    def Xs_0(self, f):
        """
        Return the surface reactance in thin film local limit at T=0K.
        Inputs: f (Hz: frequency)
        """
        #return 2*np.pi*f*self.lambL()*const.mu_0# ohms?/sq
        return self.Zs_0(f).imag# ohms?/sq
    
    def Xs(self, f, P_opt=0):
        """
        Return the surface reactance in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        #return self.Xs_0()*(1- (self.sigma2(P_opt=P_opt)-self.sigma2_0())/self.sigma2_0())# ohms?/sq
        return self.Zs(f, P_opt=P_opt).imag# ohms?/sq
    
    def Rs(self, f, P_opt=0):
        """
        Return the surface resistance in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        #return self.Xs_0()*self.sigma1(P_opt=P_opt)/self.sigma2_0()# ohms?/sq
        return self.Zs(f, P_opt=P_opt).real# ohms?/sq
        
    @property
    def Lk_0(self):
        """
        Return the kinetic inductance in thin film local limit at T=0K.
        Inputs: none
        """
        return self.nsq*const.h/(2*np.pi**2*self.tI*self.delta0*self.sigma_n)# henry?
    
    def Lk(self, f, P_opt=0):
        """
        Return the kinetic inductance in thin film local limit. (redundant)
        Inputs:  f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.Lk_0*(2*self.N0*self.delta0*self.VI/(2*self.N0*self.delta0*self.VI +self.N_qp_tot(P_opt=P_opt)*self.sigma2rat(f)))# henry?
    
    # resonance frequency
    @property
    def alpha(self):
        """
        Return the effective kinetic inductance fraction in thin film local limit.
        Inputs: none
        """
        return self.Lk_0/(self.Lg +self.Lk_0)

# old set: DO NOT USE
#    def f0(self, P_opt=0):
#        """
#        Return the resonant frequency of the resonator circuit.
#        Inputs: P_opt (Watts: Optical Power (about 1e-11))
#        """
#        return 1/np.sqrt(self.C*(self.Lk(P_opt=P_opt)+self.Lg))/(2*np.pi)# Hertz
    
#    def ffrac(self, P_opt=0):
#        """
#        Return the fractional frequency shift in resonant frequency of the resonator circuit in thin film local limit.
#        Inputs: P_opt (Watts: Optical Power (about 1e-11))
#        """
#        return self.alpha()*(self.Lk(P_opt=P_opt) - self.Lk_0())/(2*self.Lk_0())
    
#    def fnew(self, P_opt=0):
#        """
#        Return the new resonant frequency of the resonator circuit in thin film local limit.
#        Inputs: P_opt (Watts: Optical Power (about 1e-11))
#        """
#        return self.f0(P_opt=P_opt)*(1 - self.ffrac(P_opt=P_opt))
    
    @property
    def f0(self):#TODO: C and Lg are not at T=0K
        """
        Return the resonant frequency of the resonator circuit at T=0K.
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return 1/(2*np.pi*np.sqrt(self.C*(self.Lk_0+self.Lg)))# hertz
    
    def ffrac(self, f, P_opt=0):
        """
        Return the fractional frequency shift in resonant frequency of the resonator circuit in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.alpha*(self.Lk(f, P_opt=P_opt) -self.Lk_0)/(2*self.Lk_0)
    
    def fnew(self, f, P_opt=0):
        """
        Return the new resonant frequency of the resonator circuit in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.f0*(1 -self.ffrac(f, P_opt=P_opt))
    
    def fdet(self, f, P_opt=0):
        """
        Return the fractional frequency detuning in resonant frequency of the resonator circuit in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return f/self.fnew(f, P_opt=P_opt) - 1
    
    # quality factor
    def Q_qp(self, f, P_opt=0):#TODO: Q_qp = Q_i?
        """
        Return the quality factor of the resonator circuit in thin film local limit arising from quasiparticles.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.Xs_0(f)/(self.Rs(f, P_opt=P_opt)*self.alpha)
    
    def Qr(self, f, P_opt=0):
        """
        Return the resonator quality factor of the resonator circuit in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return 1/(1/self.Qc +1/self.Ql +1/self.Q_qp(f, P_opt=P_opt))

    # responsivity
    @property
    def dPabs_dPinc(self):
        """
        Returns responsivity of absorbed optical power to incident optical power.
        Inputs: none
        """
        return self.eta_opt

    @property
    def dGamma_dP(self):
        """
        Returns responsivity of quasiparticle generation rate to optical power.
        Inputs: none
        """
        return self.eta_pb/(self.delta0)

    def dN_qp_tot_dGamma(self, P_opt=0):
        """
        Returns responsivity of N_qp_tot to quasiparticle generation rate.
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return 0.5*np.sqrt(1/((self.gamma_G+(self.Gamma_opt(P_opt=P_opt)/self.VI))*self.R_eff))
    
    def dsig1_dN(self, f):
        """
        Return the responsivity of sigma1 to N_qp_tot. Uses sigma1rat and used in calculation of sigma1.
        Inputs: f (Hz: frequency)
        """
        return self.sigma2_0(f)*self.sigma1rat(f)/(2*self.N0*self.delta0*self.VI)
    
    def dsig2_dN(self, f):
        """
        Return the responsivity of sigma2 to N_qp_tot. Uses sigma2rat and used in calculation of sigma2.
        Inputs: f (Hz: frequency)
        """
        return self.sigma2_0(f)*self.sigma2rat(f)/(2*self.N0*self.delta0*self.VI)
    
    def dRs_dsig1(self, f):
        """
        Returns responsivity of surface resistance Rs to sigma1.
        Inputs: f (Hz: frequency)
        """
        return self.Xs_0(f)/self.sigma2_0(f)
    
    def dXs_dsig2(self, f):
        """
        Returns responsivity of surface reactance Xs to sigma2.
        Inputs: f (Hz: frequency)
        """
        return -self.Xs_0(f)/self.sigma2_0(f)
    
    def dlambqp_dRs(self, f):
        """
        Return the responsivity of quasiparticle loss factor lambda_qp to surface resistance Rs.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.alpha/self.Xs_0(f)
    
    def dx_dXs(self, f):
        """
        Return the responsivity of frequency detuning x to surface reactance Xs.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.alpha/(2*self.Xs_0(f))
    
    def dQqp_dRs(self, f, P_opt=0):
        """
        Return the responsivity of quasiparticle quality factor Q_qp to  surface resistance Rs.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return -self.Q_qp(f, P_opt=P_opt)**2*self.alpha/self.Xs_0(f)

    def dx_dP(self, f, P_opt=0):
        """
        Return the responsivity of frequency detuning x to optical power.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.dx_dXs(f)*self.dXs_dsig2(f)*self.dsig2_dN(f)*self.dN_qp_tot_dGamma(P_opt=P_opt)*self.dGamma_dP*self.dPabs_dPinc
    
    def dQqp_dP(self, f, P_opt=0):
        """
        Return the responsivity of quasiparticle quality factor Q_qp to optical power.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return self.dQqp_dRs(f)*self.dRs_dsig1(f)*self.dsig1_dN(f)*self.dN_qp_tot_dGamma(P_opt=P_opt)*self.dGamma_dP*self.dPabs_dPinc
    
    # s21
    def S21(self, f, P_opt=0):#TODO: rewrite?
        """
        Return the resonator quality factor of the resonator circuit in thin film local limit.
        Inputs: f (Hz: frequency), P_opt (Watts: Optical Power (about 1e-11))
        """
        return 1 -self.Qr(f, P_opt=P_opt)/(self.Qc*(1+2j*self.Qr(f, P_opt=P_opt)*self.fdet(f, P_opt=P_opt)))
    
    # noise
    def nep_phot(self, P_opt=0):
        """
        Return the noise equivalent power (NEP/W/sqrt(Hz)) of photon noise.
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return np.sqrt(2.*(const.h*self.nu_opt*P_opt + P_opt**2/self.delta_nuopt))
    
    def nep_rec(self, P_opt=0):
        """
        Return the noise equivalent power (NEP/W/sqrt(Hz)) of recombination noise due to incident optical power.
        Inputs: P_opt (Watts: Optical Power (about 1e-11))
        """
        return 2*np.sqrt(self.delta0*P_opt/(self.eta_pb*self.eta_opt))
    
#    def nep_tls(self, P_opt=0):
#        """
#        Return the noise equivalent power (NEP/W/sqrt(Hz)) of 2-level system noise.
#        Inputs: P_opt (Watts: Optical Power (about 1e-11))
#        """
#        return np.sqrt(2.*const.h*self.nu_opt()*P_opt + (P_opt**2)/(self.delta_nuopt))
    
    # finding P_opt from fnew
    def P_opt_r(self, fnew):#TODO: REWRITE
        """
        Return P_opt value from fnew.
        Inputs: fnew
        """
        Lk=2*(self.Lg +self.Lk_0)*(1 -2*np.pi*fnew*np.sqrt(self.C*(self.Lg +self.Lk_0))) +self.Lk_0
        Nqp=-2*self.N0*self.delta0*self.VI*(1 +self.Lk_0/Lk)/self.sigma2rat
        return (Nqp**2*self.R_eff/self.VI -self.gamma_G)*self.delta/(self.eta_pb*self.dPabs_dPinc)
    