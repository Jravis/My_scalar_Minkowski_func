
#############################################################################
#Copyright (c) 2017, Sandeep Rana & Tuhin Ghosh
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################

"""
Code for testing non-Gaussian
features in a map using scalar
Mikowski Functionals S0, S1, S2
See "Minkowski functionals used in
the morphological analysis of cosmic
microwave background anisotropy maps"
Jens Schmalzing et. al. 1998
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

#++++++++++++++++++++ l Filter type 1 ++++++++++++++++++++++
"""
In this section I am Generating filter function
anchoring at diffetent l0 value such that
if l<= l0, f = 0
if l> l0 , f = 1
Filter function written in a way that it won't go
zero sharply but rather in a smooth fashion
"""
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

window_func_filter1 = np.zeros((len(l0), nlmax), dtype=np.float)

for i in xrange(len(l0)):
    ini = l0[i]
    window_func_filter1[i, :] = filter_arr1(ini, nlmax)

#++++++++++++++++++++ end l Filter ++++++++++++++++++++++

#++++++++++++++++++++ l Filter type 2 ++++++++++++++++++++++
"""
In this section I am Generating filter function
anchoring at diffetent l0 value such that
if l<= l0, f = 0
if l> l0 , f = 1
Filter function written in a way that it won't go
zero sharply but rather in a smooth fashion
"""
l0 = [30, 50 , 70, 90]

def filter_arr2(ini, final):
    """
    tan(x) filter
    """

    delta_l = 5
    window_func = np.zeros(nlmax, dtype=np.float)

    #for l in xrange(ini, final):
    for l in xrange(0, final):
#        if ini + delta_l <= l :
#            window_func[l] = 1.0
#        elif ini <= l < ini + delta_l:
#            window_func[l] = (np.tanh(np.pi*(l-ini) / delta_l))
        window_func[l] = 0.5*(1.+np.tanh((l-ini) / (delta_l*1.0)))
    return window_func


window_func_filter2 = np.zeros((len(l0), nlmax), dtype=np.float)

for i in xrange(len(l0)):
    ini = l0[i]
    window_func_filter2[i, :] = filter_arr2(ini, nlmax)

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


def Map_Prep(inp_map, Sky_mask, lFilter, indices, bmask):
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

    inp_map = hp.alm2map(hp.almxfl(inp_map_alm,lFilter), Nside, verbose=False)

    temp_inp_map = inp_map[indices]

    inp_map_mean = np.mean(temp_inp_map)
    inp_map_sigma = np.std(temp_inp_map)


    inp_map *=bmask
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
        if inp_mask[ipix] > 0.9:
            binary_mask[ipix]=1


    return binary_mask

#====================================================================

def compute_minkowski(Map, sky_mask, binary_temp_mask, fn):
    """
    """

    delta = 0.2
    nu = np.arange(-4, 4., delta)

    S0 = np.zeros( (5, len(nu)))
    S1 = np.zeros( (5, len(nu)))
    S2 = np.zeros( (5, len(nu)))
#    Ana2 = np.zeros((5,len(nu)-1))

    ind = (binary_temp_mask==1)
    NPIX = binary_temp_mask[ind]
    NPIX = len(NPIX)

    temp_mask = np.zeros(len(binary_temp_mask))
    indd = (binary_temp_mask > 0)
    temp_mask[indd]= 1
    indd = (binary_temp_mask <= 0)
    temp_mask[indd]= -9999

    for  l in xrange(0, 4):

        u, grad_u, kapa_u = Map_Prep(Map, sky_mask, window_func_filter2[l, :], ind, binary_temp_mask)

        u     *= binary_temp_mask
        grad_u *= binary_temp_mask
        kapa_u *= binary_temp_mask

        for j in xrange(len(nu)):

#+++++++Analytic part+++++++++++++++

#            temp = grad_u**2*4
#            mean_ana = np.mean(u)
#            Sig_Sum = (np.mean(u**2) -np.mean(u)**2 )
#            tau_Sum = 0.5*(temp)
#            nu_mean = (nu[j+1]+nu[j])/2.
#            Ana2[l, j] = analaytic_S2(mean_ana, nu_mean, tau_Sum, Sig_Sum)

#+++++++ Data part+++++++++++++++


            valid_inices = (temp_mask!=-9999)
            u1 = u[valid_inices]
            index1 = (u1 > nu[j])
            temp1 = u1[index1]
            S0[l, j] = len(temp1)/(NPIX*1.0)

            #index = (u  > nu[j])*(u  < nu[j+1])

            index = (u  > nu[j]-delta*0.5) * (u  < nu[j]+delta*0.5) # Half bining

            temp2 = grad_u[index]
            temp3 = kapa_u[index]

            S1[l, j] = (np.sum(temp2)/delta)/(NPIX*1.0)
            S2[l, j] = (np.sum(temp3)/delta)/(NPIX*1.0)



    fname1 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S0_%d.txt' % fn
    fname2 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S1_%d.txt' % fn
    fname3 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S2_%d.txt' % fn

    np.savetxt(fname1, zip(S0[0,:], S0[1,:], S0[2,:], S0[3,:], S0[4,:]),
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e',
                    delimiter='\t', header='l_10\tl_20\tl_40\tl_80\tl_120')


    np.savetxt(fname2, zip(S1[0,:], S1[1,:], S1[2,:], S1[3,:], S1[4,:]),
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e',
                    delimiter='\t', header='l_10\tl_20\tl_40\tl_80\tl_120')


    np.savetxt(fname3, zip(S2[0,:], S2[1,:], S2[2,:], S2[3,:], S2[4,:]),
                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e',
                    delimiter='\t', header='l_10\tl_20\tl_40\tl_80\tl_120')


#    fname1 = 'Gauss_haslam_Minkowski_functional_Anal2_%d.txt' % fn
#    np.save(fname1, zip(Ana2[0,:], Ana2[1,:], Ana2[2,:], Ana2[3,:], Ana2[4,:]),
#                    delimiter='\t',
#                    fmt='%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e'
#                    , header='l_10\tl_20\tl_40\tl_80\t_l_120')


#====================================================================

def parllel_compute(nmin, nmax, sky_Mask, b_mask):


    for nn in xrange(nmin, nmax):
        filename = 'Gaussian_25K_Maps/haslam_gaussMap_25K_%d.fits'%nn
        compute_minkowski(hp.read_map(filename), sky_Mask, b_mask, nn)

#====================================================================

def main():

    TEMP = '25K'

    #name = '/jbodstorage/data_sandeep/sandeep/Bispectrum_data/input_Maps/Mask_80K_apod_300arcm_ns_128.fits'
    name = '../Mask_80K_apod_300arcm_ns_128.fits'
    Mask_80K = hp.fitsfunc.read_map(name, verbose=False)

    #f_name1 = "/jbodstorage/data_sandeep/sandeep/Bispectrum_data/input_Maps/Mask_%s_apod_300arcm_ns_128.fits" % TEMP
    f_name1 = "../Mask_%s_apod_300arcm_ns_128.fits" % TEMP
    print f_name1
    ap_mask_128 = hp.fitsfunc.read_map(f_name1, verbose=False)
    b_mask = thresh_masking(ap_mask_128)

    max_core = 1
    count=0
    increment = 100
    jobs = []

    for i in xrange(0, max_core):
        nmin = count
        nmax = count + increment
        if nmax == 100:
            nmax = 101
        s = Process(target=parllel_compute, args=(nmin, nmax, Mask_80K, b_mask))
        jobs.append(s)
        s.start()
        count = nmax

    for s in jobs:
        s.join()


if __name__ == "__main__":
    main()







