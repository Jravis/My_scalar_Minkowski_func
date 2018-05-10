import numpy as np
import matplotlib.pyplot as plt

#++++++++++++++++This one to check filter for Bispectrum case ++++++++++++
lmax = 256
nbin = 11
NSIDE = 128


#index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)



index = [[10, 30], [25, 45], [40, 60],[55, 75], [70, 90], [85, 105], [100, 120], [115, 135], [130, 150], [145, 165],[160, 180]]
index = np.asarray(index)
print index.shape

def filter_arr(ini, final):

    delta_l=5

    window_func = np.zeros(lmax, dtype=np.float)
    for l in xrange(ini, final):
        if ini + delta_l <= l <= final - delta_l:
            window_func[l] = 1.0

        elif ini < l < ini + delta_l:
            window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2
        elif final - delta_l < l < final:
            window_func[l] = 1.0 * np.cos(np.pi * 0.5 * (l - (final-delta_l)) / delta_l) ** 2
#        print ini, final, l, window_func[l]
    return window_func

window_func_filter = np.zeros((nbin, lmax), dtype=np.float)

for i in xrange(0, nbin):
    ini = index[i, 0]
    final = index[i, 1]
    window_func_filter[i, :] = filter_arr(ini, final)




plt.figure(1)
plt.plot(window_func_filter[0,:])
plt.plot(window_func_filter[1,:])
plt.plot(window_func_filter[2,:])
plt.plot(window_func_filter[3,:])
plt.plot(window_func_filter[4,:])
plt.xlim(9, 90)



#++++++++++++++++This one to check filter for minkowski case++++++++++++

l0 = [10, 30, 50 , 70, 90]

nlmax = 256

def filter_arr1(ini, final):
    """
    Cos^2 filter
    """
    delta_l = 20
    window_func = np.zeros(nlmax, dtype=np.float)
    for l in xrange(ini, final):
        if ini + delta_l <= l :
            window_func[l] = 1.0

        elif ini <= l < ini + delta_l:
            window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2

    return window_func



def filter_arr2(ini, final):
    """
    tan(x) filter
    """

    delta_l = 5
    window_func = np.zeros(nlmax, dtype=np.float)

    #for l in xrange(ini, final):
    print ini
    for l in xrange(0, final):
        #if ini + delta_l <= l :
        #    window_func[l] = 1.0
        #elif ini <= l < ini + delta_l:
        #    window_func[l] = (np.tanh(np.pi*(l-ini) / delta_l))

        window_func[l] = 0.5*(1+np.tanh((l-ini) / (delta_l*1.)))

    return window_func


window_func_filter = np.zeros((len(l0), nlmax), dtype=np.float)

for i in xrange(len(l0)):
    ini = l0[i]
    window_func_filter[i, :] = filter_arr2(ini, nlmax)

#plt.style.use('classic')


plt.figure(2, figsize=(8, 6))
plt.plot(window_func_filter[0,:], linestyle='-', linewidth=1.8)
plt.plot(window_func_filter[1,:], linestyle='-', linewidth=1.8)
plt.plot(window_func_filter[2,:], linestyle='-', linewidth=1.8)
plt.plot(window_func_filter[3,:], linestyle='-', linewidth=1.8)
plt.plot(window_func_filter[4,:], linestyle='-', linewidth=1.8)

plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$f_{\ell}$', fontsize=16)
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
plt.tight_layout()
plt.xlim(0, 130)
plt.ylim(-0.01, 1.1)
plt.savefig("New_filter_prava_filter.pdf", dpi=600)




plt.figure(3, figsize=(8, 6))
plt.plot(window_func_filter[0,:], linestyle='-', linewidth=1.8)
plt.axvline(x=20, linestyle='-.', linewidth=1.5, color='k')
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$f_{\ell}$', fontsize=16)
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
plt.tight_layout()
plt.xlim(7, 30)
plt.ylim(-0.1, 1.1)
plt.savefig("New_filter_zoom_prava_filter.pdf", dpi=600)

window_func=[]
for l in xrange(0, 30):
    window_func.append(0.5*(1+np.tanh((l-20) / 5.)))
plt.figure(4, figsize=(8, 6))
plt.plot(window_func, linestyle='-', linewidth=1.8)





plt.show()

