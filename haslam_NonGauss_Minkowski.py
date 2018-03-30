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
from  multiprocessing import Process
from scipy import special


# Global values
nlmax = 256
Nside =128
Npix = hp.nside2npix(Nside)
cosbysin = np.zeros(Npix)
for ipix in xrange(Npix):
    theta, phi = hp.pix2ang(Nside, ipix)
    cosbysin[ipix] = np.cos(theta)/np.sin(theta)


#++++++++++++++++++++ l Filter ++++++++++++++++++++++
"""
In this section I am Generating filter function
anchoring at diffetent l0 value such that
if l<= l0, f = 0
if l> l0 , f = 1
Filter function written in a way that it won't go
zero sharply but rather in a smooth fashion
"""
#l0 = [10, 20, 40 , 80, 120]
l0 = [10, 30, 50 , 70, 90]

def filter_arr1(ini, final):
    delta_l = 10
    window_func = np.zeros(nlmax, dtype=np.float)
    for l in xrange(ini, final):
        if ini + delta_l <= l :
            window_func[l] = 1.0

        elif ini <= l < ini + delta_l:
            window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2

    return window_func

window_func_filter = np.zeros((len(l0), nlmax), dtype=np.float)

for i in xrange(len(l0)):
    ini = l0[i]
    window_func_filter[i, :] = filter_arr1(ini, nlmax)

#++++++++++++++++++++ end l Filter ++++++++++++++++++++++

#++++++++++++++ Analytical Form of Scalar Minkowski ++++++

def analaytic_S0(mean_g, x, Sigma):

    return 0.5*special.erfc( (x - mean_g) / np.sqrt(2.*Sigma) )

def analaytic_S1(mean_g, x, tau, Sigma):

    temp =  np.sqrt(tau/Sigma) * (1./8.)

    return temp * np.exp(- ((x - mean_g)**2.) /2/Sigma)

def analaytic_S2(mean_g, x, tau, Sigma):

    temp =  (tau/Sigma) * (x - mean_g)/np.sqrt(2*Sigma) * (1./ (2.*np.pi**1.5))

    return temp * np.exp(- ((x - mean_g)**2.) /2/Sigma)

#++++++++++++++ End analytical Form of Scalar Minkowski ++++++


def Map_Prep(inp_map, Sky_mask, lFilter):
    """
    # Map prepration
    param1 :- Input Map
    param2 :- galactic Mask
    returns
    1 : Scalar u field
    2 : grad|u|
    3 : grad|u|*Kappa
    """

    inp_map = inp_map*Sky_mask
    inp_map_alm = hp.map2alm(inp_map, lmax=nlmax)

    inp_map = hp.alm2map(hp.almxfl(inp_map_alm,lFilter), Nside)

    inp_map_mean = np.mean(inp_map)
    inp_map_sigma = np.std(inp_map)

    g_map = (inp_map-inp_map_mean)/inp_map_sigma # we calling it g_map where in Jens Schmalzing et. al. 1998 its u(generic scalar field)

    g_alm = hp.sphtfunc.map2alm(g_map, lmax=nlmax)
    g_map, d_theta_g, d_phi_g = hp.sphtfunc.alm2map_der1(g_alm, Nside)


    #Gradient of gmap grad|g|
    grad_g_map = np.sqrt(d_theta_g**2 + d_phi_g**2)/4

    # Computing Kappa
    u1 = d_theta_g
    u2 = d_phi_g


    d_theta_g_alm = hp.sphtfunc.map2alm(d_theta_g)
    d_phi_g_alm = hp.sphtfunc.map2alm(d_phi_g)

    d_phi_g, Du_theta_phi_g, Du_phi_phi_g = hp.sphtfunc.alm2map_der1(d_phi_g_alm, Nside)
    d_theta_g, u11, Du_theta_phi_g = hp.sphtfunc.alm2map_der1(d_theta_g_alm, Nside)

    u12 = Du_theta_phi_g - cosbysin * d_phi_g

    u22 = Du_phi_phi_g + cosbysin* d_theta_g

    kapa = (2* u1 * u2* u12 - u1**2 * u22 - u2**2 * u11)
    kapa /= (u1**2 + u2**2)
    kapa /= 2*np.pi

    return g_map, grad_g_map, kapa

#=================================================================

def thresh_masking(inp_mask):
    """
    param:-  Input Mask(apodize)
    return:- binary mask with some threshold value
    chosen fby us in this case I am taking 0.998
    """
    binary_mask = np.zeros(len(inp_mask))

    for ipix in xrange(len(inp_mask)):
        if inp_mask[ipix] > 0.998:
            binary_mask[ipix]=1


    return binary_mask

#====================================================================

def compute_minkowski(Map, sky_mask, binary_temp_mask, fn):
    """
    """

    delta = 0.2
    nu = np.arange(-4, 4., delta)

    S0 = np.zeros( (5, len(nu)-1))
    S1 = np.zeros( (5, len(nu)-1))
    S2 = np.zeros( (5, len(nu)-1))
#    Ana2 = np.zeros((5,len(nu)-1))

    ind = (binary_temp_mask==1)
    NPIX = binary_temp_mask[ind]
    NPIX = len(NPIX)


    for  l in xrange(0, 4):

        u, grad_u, kapa_u = Map_Prep(Map, sky_mask, window_func_filter[l, :])

        u     *= binary_temp_mask
        grad_u *= binary_temp_mask
        kapa_u *= binary_temp_mask

        for j in xrange(len(nu)-1):

#+++++++Analytic part+++++++++++++++

#            temp = grad_u**2*4
#            mean_ana = np.mean(u)
#            Sig_Sum = (np.mean(u**2) -np.mean(u)**2 )
#            tau_Sum = 0.5*(temp)
#            nu_mean = (nu[j+1]+nu[j])/2.
#            Ana2[l, j] = analaytic_S2(mean_ana, nu_mean, tau_Sum, Sig_Sum)

#+++++++ Data part+++++++++++++++

            index = (u  > nu[j])*(u  < nu[j+1])
            index1 = (u>nu[i])
            temp1 = u[index1]
            temp2 = grad_u[index]
            temp3 = kapa_u[index]

            S0[l, j] = (len(temp1))/(NPIX*1.0)
            S1[l, j] = (np.sum(temp2)/delta)/(NPIX*1.0)
            S2[l, j] = (np.sum(temp3)/delta)/(NPIX*1.0)



    fname1 = 'Gauss_haslam_Minkowski_functional_S0_%d.txt' % fn
    fname2 = 'Gauss_haslam_Minkowski_functional_S1_%d.txt' % fn
    fname3 = 'Gauss_haslam_Minkowski_functional_S2_%d.txt' % fn

    np.save(fname1, zip(S0[0,:], S0[1,:], S0[2,:], S0[3,:], S0[4,:]),
                    delimiter='\t',
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e'
                    , header='l_10\tl_20\tl_40\tl_80\tl_120')


    np.save(fname2, zip(S1[0,:], S1[1,:], S1[2,:], S1[3,:], S1[4,:]),
                    delimiter='\t',
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e'
                    , header='l_10\tl_20\tl_40\tl_80\tl_120')


    np.save(fname3, zip(S2[0,:], S2[1,:], S2[2,:], S2[3,:], S2[4,:]),
                    delimiter='\t',
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e'
                    , header='l_10\tl_20\tl_40\tl_80\tl_120')


#    fname1 = 'Gauss_haslam_Minkowski_functional_Anal2_%d.txt' % fn
#    np.save(fname1, zip(Ana2[0,:], Ana2[1,:], Ana2[2,:], Ana2[3,:], Ana2[4,:]),
#                    delimiter='\t',
#                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e'
#                    , header='l_10\tl_20\tl_40\tl_80\t_l_120')


#====================================================================

def parllel_compute(nmin, nmax, sky_Mask, b_mask):


    for nn in xrange(nmin, nmax):
        filename = '%d.fits'%nn
        compute_minkowski(hp.read_map(filename), sky_Mask, b_mask, nn)

#====================================================================

def main():

    TEMP = '25K'

    name = '/jbodstorage/data_sandeep/sandeep/Bispectrum_data/input_Maps/Mask_80K_apod_300arcm_ns_128.fits'
    Mask_80K = hp.fitsfunc.read_map(name, verbose=False)

    f_name1 = "/jbodstorage/data_sandeep/sandeep/Bispectrum_data/input_Maps/Mask_%s_apod_300arcm_ns_128.fits" % TEMP
    print f_name1
    ap_mask_128 = hp.fitsfunc.read_map(name, verbose=False)
    b_mask = thresh_masking(ap_mask_128)

    max_core = 25
    count=0
    increment = 40
    jobs = []

    for i in xrange(0, max_core):
        nmin = count
        nmax = count + increment
        if nmax == 1000:
            nmax = 1001
        s = Process(target=parllel_compute, args=(nmin, nmax, Mask_80K, b_mask))
        jobs.append(s)
        s.start()
        count = nmax

    for s in jobs:
        s.join()


if __name__ == "__main__":
    main()







