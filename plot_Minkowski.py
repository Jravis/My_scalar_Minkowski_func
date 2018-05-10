import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




l0 = [ 30, 50 , 70, 90]
#l0 = [1., 3., 5. , 7., 9.]
delta = 0.2
nu = np.arange(-4, 4., delta)

S0_data = np.genfromtxt('New_filter/haslam_25K_Minkowski_functional_S0.txt', usecols=4, delimiter='\t')


Data_S = np.zeros( (4,3, len(S0_data)) )

foo = ['S0', 'S1', 'S2']
for l in xrange(0, 4):
    for i in xrange(0, 3):
        fname = 'New_filter/haslam_25K_Minkowski_functional_%s.txt'% foo[i]
        Data_S[l,i,:] = np.genfromtxt(fname, usecols=l, delimiter='\t')


gauss_S0 = np.zeros((100, len(S0_data)) )
gauss_S1 = np.zeros((100, len(S0_data)) )
gauss_S2 = np.zeros((100, len(S0_data)) )


gauss_S_mean = np.zeros((4, 3, len(S0_data)))
gauss_S_std = np.zeros((4, 3, len(S0_data)))


for l in xrange(0, 4):
    for nn in xrange(0, 100):

        s0 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S0_%d.txt' % nn
        s1 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S1_%d.txt' % nn
        s2 = 'New_filter/Mink_Gaussian_New_25K/Gauss_haslam_Minkowski_functional_S2_%d.txt' % nn
        gauss_S0[nn,:] = np.genfromtxt(s0, usecols=l, delimiter='\t')
        gauss_S1[nn,:] = np.genfromtxt(s1, usecols=l, delimiter='\t')
        gauss_S2[nn,:] = np.genfromtxt(s2, usecols=l, delimiter='\t')

    gauss_S_mean[l,0, :] = np.mean(gauss_S0, axis=0)
    gauss_S_mean[l,1,:] = np.mean(gauss_S1, axis=0)
    gauss_S_mean[l,2,:] = np.mean(gauss_S2, axis=0)

    gauss_S_std[l,0, :] = np.std(gauss_S0, axis=0)
    gauss_S_std[l,1, :] = np.std(gauss_S1, axis=0)
    gauss_S_std[l,2, :] = np.std(gauss_S2, axis=0)



plt.style.use("classic")

fig = plt.figure(3, figsize=(24, 24))
gs = gridspec.GridSpec(4, 3)

for Ix in xrange(0, 4):
    for Iy in xrange(0, 3):
        ax1 = plt.subplot(gs[Ix, Iy])
        temp = (Data_S[Ix, Iy,:]-gauss_S_mean[Ix, Iy,:])/ np.max(np.abs(gauss_S_mean[Ix, Iy,:]))
        std =  np.std(temp)
        #ax1.plot(nu[:-1], l0[Ix]*temp, marker='s', mfc='none', mew=2, mec='c', color='c', linewidth=2, linestyle='-', label='$l_{0}=%d$'%l0[Ix])

        ax1.plot(nu, temp, marker='s', mfc='none', mew=2, mec='#01665e', color='#01665e', linewidth=2, linestyle='-', label='$l_{0}=%d$'%l0[Ix])

        ax1.fill_between(nu, (temp-std) , (temp+std) , edgecolor='k', facecolor='#525252', alpha=0.5)
        ax1.fill_between(nu, (temp-2.*std) , (temp+2.*std) , edgecolor='k', facecolor='#525252', alpha=0.5)
        ax1.fill_between(nu, (temp-3.*std) , (temp+3.*std) , edgecolor='k', facecolor='#525252', alpha=0.5)

        if Iy==0:
            ax1.set_ylabel(r"$ (V_{0}^{Data}-V_{0}^{Mean})/V_{0}^{G,max}$", fontsize=18)
        if Iy==1:
            ax1.set_ylabel(r"$ (V_{1}^{Data}-V_{1}^{Mean})/V_{1}^{G,max}$", fontsize=18)
        if Iy==2:
            ax1.set_ylabel(r"$ (V_{2}^{Data}-V_{2}^{Mean})/V_{2}^{G,max}$", fontsize=18)

        ax1.legend(loc=1, fontsize=18, frameon=False)
        ax1.set_xlabel(r"$\nu$", fontsize=18)
        ax1.set_xlim(-4, 4)
        plt.minorticks_on()
        plt.legend(fontsize=10, frameon=False)
        plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
        plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)


plt.savefig("New_filter/Minkowski_all_l0_25K_1.pdf", dpi=600)

fig = plt.figure(4, figsize=(24, 24))
gs = gridspec.GridSpec(4, 3)

for Ix in xrange(0, 4):
    for Iy in xrange(0, 3):
        ax1 = plt.subplot(gs[Ix, Iy])

        ax1.plot(nu, Data_S[Ix, Iy,:], marker='s', mfc='none',mew=2,mec='peru' , color='peru', linewidth=2, linestyle='-', label='Data $l_{0}=%d$'%l0[Ix])
        ax1.plot(nu, gauss_S_mean[Ix, Iy,:], marker='s', mfc='none',mew=2, mec='#01665e', color='#01665e', linewidth=2, linestyle='-', label='Gauss Mean $l_{0}=%d$'%l0[Ix])

        ax1.fill_between(nu, gauss_S_mean[Ix, Iy,:]-gauss_S_std[Ix, Iy,:] ,gauss_S_mean[Ix, Iy,:]+gauss_S_std[Ix, Iy,:], alpha=0.5
                            , edgecolor='k', facecolor='#525252')

        ax1.fill_between(nu, gauss_S_mean[Ix, Iy,:]-2.*gauss_S_std[Ix, Iy,:] ,gauss_S_mean[Ix, Iy,:]+2.*gauss_S_std[Ix, Iy,:], alpha=0.5
                            , edgecolor='k', facecolor='#737373')

        ax1.fill_between(nu, gauss_S_mean[Ix, Iy,:]-3.*gauss_S_std[Ix, Iy,:] ,gauss_S_mean[Ix, Iy,:]+3.*gauss_S_std[Ix, Iy,:],alpha=0.5
                            , edgecolor='k', facecolor='#969696')

        if Iy==0:
            ax1.set_ylabel(r"$ V_{0} $", fontsize=18)
            plt.legend(loc=3, fontsize=18, frameon=False)
        if Iy==1:
            ax1.set_ylabel(r"$ V_{1}$", fontsize=18)
            plt.legend(loc=8, fontsize=18, frameon=False)
        if Iy==2:
            ax1.set_ylabel(r"$ V_{2}$", fontsize=18)
            plt.legend(loc=4, fontsize=18, frameon=False)

        ax1.set_xlabel(r"$\nu$", fontsize=18)
        ax1.set_xlim(-3, 3)
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
        plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.savefig("New_filter/Minkowski_all_l0_25K.pdf", dpi=600)


plt.show()



