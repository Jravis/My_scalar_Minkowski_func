"""
Code for testing non-Gaussian
features in a map using scalar
Mikowski Functionals S0, S1, S2
See "Minkowski functionals used in
the morphological analysis of cosmic
microwave background anisotropy maps"
Jens Schmalzing et. al. 1998

@Author Sandeep Rana & Tuhin Ghosh
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import special


#++++++++++++++++++++ l Filter ++++++++++++++++++++++
"""
In this section I am Generating

"""


































Nside =512

cl = hp.fitsfunc.read_cl("cl.fits")

gauss_map = hp.synfast(cl[0], 512, lmax=3*Nside-1, fwhm=0, pixwin=True, verbose=False, pol=False)

def analaytic_S2(mean_g, x, tau, Sigma):

    temp =  (tau/Sigma) * (x - mean_g)/np.sqrt(2*Sigma) * (1./ (2.*np.pi**1.5))

    return temp * np.exp(- ((x - mean_g)**2.) /2/Sigma)

def analaytic_S1(mean_g, x, tau, Sigma):

    temp =  np.sqrt(tau/Sigma) * (1./8.)

    return temp * np.exp(- ((x - mean_g)**2.) /2/Sigma)


def analaytic_S0(mean_g, x, Sigma):

    return 0.5*special.erfc( (x - mean_g) / np.sqrt(2.*Sigma) )



#+++++++++++++++++++++++++++++++++++++++++++

G_mean = np.mean(gauss_map)
sigma = np.std(gauss_map)
Npix = hp.nside2npix(Nside)

#print Npix


#================================================================dd
g_map = (gauss_map-G_mean)/sigma

#Cl = hp.anafast(g_map, lmax=3*Nside-1)
#ell = np.arange(len(Cl))


cosbysin = np.zeros(Npix)
for ipix in xrange(Npix):
    theta, phi = hp.pix2ang(Nside, ipix)
    cosbysin[ipix] = np.cos(theta)/np.sin(theta)



g_alm = hp.sphtfunc.map2alm(g_map, lmax=3*Nside-1)
g_map, d_theta, d_phi = hp.sphtfunc.alm2map_der1(g_alm, Nside)



grad_gauss_map = np.sqrt(d_theta**2 + d_phi**2)/4

u1 = d_theta
u2 = d_phi


d_theta_alm = hp.sphtfunc.map2alm(d_theta)
d_phi_alm = hp.sphtfunc.map2alm(d_phi)

d_phi, Du_theta_phi, Du_phi_phi = hp.sphtfunc.alm2map_der1(d_phi_alm, Nside)
d_theta, u11, Du_theta_phi = hp.sphtfunc.alm2map_der1(d_theta_alm, Nside)

u12 = Du_theta_phi - cosbysin * d_phi
u22 = Du_phi_phi + cosbysin* d_theta

kapa = (2* u1 * u2* u12 - u1**2 * u22 - u2**2 * u11)
kapa /= (u1**2 + u2**2)
kapa /= 2*np.pi


#=================================================================

mean_ana = np.mean(g_map)
Sig_Sum = (np.mean(g_map**2) -np.mean(g_map)**2 )
tau_Sum = 0.5*(np.mean(d_theta*d_theta + d_phi*d_phi))

#Sig_Sum = 0.
#tau_Sum = 0.
#for i in xrange(len(Cl)):
#     Sig_Sum = Sig_Sum + (2*ell[i]+1)*Cl[i]
#     tau_Sum = tau_Sum + (2*ell[i]+1)*Cl[i] * (0.5*ell[i]*(ell[i]+1))
#print SigmaO, Sigma1

delta = 0.2
nu = np.arange(-4, 4., delta)

SO = np.zeros(len(nu)-1)
S1 = np.zeros(len(nu)-1)
S2 = np.zeros(len(nu)-1)
Ana2 = np.zeros(len(nu)-1)
Ana1 = np.zeros(len(nu)-1)
Ana0 = np.zeros(len(nu)-1)

for i in xrange(len(nu)-1):

    index = (g_map>nu[i])*(g_map<nu[i+1])
    index1 = (g_map>nu[i])

    temp1 = g_map[index1]
    temp2 = grad_gauss_map[index]
    temp3 = kapa[index]

    nu_mean = (nu[i+1]+nu[i])/2.
    Ana0[i] = analaytic_S0(mean_ana, nu_mean, Sig_Sum)
    Ana1[i] = analaytic_S1(mean_ana, nu_mean, tau_Sum, Sig_Sum)
    Ana2[i] = analaytic_S2(mean_ana, nu_mean, tau_Sum, Sig_Sum)

    SO[i] = (len(temp1))/(Npix*1.0)
    S1[i] = (np.sum(temp2)/delta)/(Npix*1.0)
    S2[i] = (np.sum(temp3)/delta)/(Npix*1.0)

plt.style.use('classic')

plt.figure(1, figsize=(8, 6))
plt.plot(nu[:-1], SO, 'o' , color='orange', alpha=0.8, label=r'$Data$')
plt.plot(nu[:-1], Ana0, linestyle='-' , color='orange',linewidth=3, alpha=0.5, label=r'$Analytical$')

plt.legend(loc=2, fontsize=14, frameon=False)
plt.minorticks_on()
plt.ylim(-0.001, 1.05)
plt.xlabel(r'$S_{1}$', fontsize=14)
plt.ylabel(r'$\nu$', fontsize=14)
plt.tick_params(axis='both', which='minor', length=4, width=2, labelsize=12)
plt.tick_params(axis='both', which='major', length=8, width=3, labelsize=12)
plt.tick_params(which='major', length=8, width=3, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, top='on')
plt.tick_params(which='major', length=8, width=3, labelsize=12, top='on')
plt.tight_layout()

plt.savefig("S0.png", dpi=800)




plt.figure(2, figsize=(8, 6))
plt.plot(nu[:-1], S1, 'b o' , color='teal', alpha=0.8, label=r'$Data$')
plt.plot(nu[:-1], Ana1, linestyle='-' , color='g',linewidth=3, alpha=0.5, label=r'$Analytical$')
plt.legend(loc=2, fontsize=14, frameon=False)
plt.minorticks_on()
plt.xlabel(r'$S_{1}$', fontsize=14)
plt.ylabel(r'$\nu$', fontsize=14)
plt.tick_params(axis='both', which='minor', length=4, width=2, labelsize=12)
plt.tick_params(axis='both', which='major', length=8, width=3, labelsize=12)
plt.tick_params(which='major', length=8, width=3, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, top='on')
plt.tick_params(which='major', length=8, width=3, labelsize=12, top='on')
plt.tight_layout()


plt.savefig("S1.png", dpi=800)


plt.figure(3, figsize=(8, 6))
plt.plot(nu[:-1], S2, 'b o' , color='crimson', alpha=0.8, label=r'Data')
plt.plot(nu[:-1], Ana2, linestyle='-' , color='crimson', linewidth=3, alpha=0.5, label=r'Analytical')
plt.legend(loc=2, fontsize=14, frameon=False)
plt.minorticks_on()
plt.xlabel(r'$S_{2}$', fontsize=14)
plt.ylabel(r'$\nu$', fontsize=14)
plt.tick_params(axis='both', which='minor', length=4, width=2, labelsize=12)
plt.tick_params(axis='both', which='major', length=8, width=3, labelsize=12)
plt.tick_params(which='major', length=8, width=3, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, right='on')
plt.tick_params(which='minor', length=4, width=2, labelsize=12, top='on')
plt.tick_params(which='major', length=8, width=3, labelsize=12, top='on')
plt.tight_layout()

plt.savefig("S2.png", dpi=800)



plt.show()
