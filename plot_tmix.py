

#Custom Modules
import CCAW
import nibabel

#Scikit Modules
from sklearn.cluster import KMeans

#Pymix Modules
from pymix.distributions.student import MultivariateTDistribution as TD
from pymix.models.mixture import MixtureModel as mixture
from pymix.util.dataset import DataSet

#Other
import numpy as np
import numpy.random as npr
from scipy import stats
import pylab as pl
from matplotlib.offsetbox import AnchoredText

# def myEStep(data, mix_posterior, mix_pi, EStepParam):

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]

def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[1,0,0], a=1, lw=2,component=0,pi=0.):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)
    ax.add_artist(ellip)

max_comp = 2
output = {}
output_arr = []
for n in range(max_comp,max_comp+1,1):
    print n,' nd/th iteration'
    #Configuration
    n_components = n #Number of Mixture components
    n_dim = 2 #Number of Dimensions
    max_iter = 10000
    tol=1e-3

    #Reading data
    # x_image_pname = 'PE_m.nii.gz'
    # x_mask_pname =  'mask_wm.nii.gz'  #self masked
    # y_image_pname = 'PD_m.nii.gz'
    # y_mask_pname =  'mask_wm.nii.gz'
    # X = CCAW.BaseData(x_image_pname, x_mask_pname)
    # Y = CCAW.BaseData(y_image_pname, y_mask_pname)
    # hb = pl.hexbin(X.image_data_masked, Y.image_data_masked, gridsize=50, bins='log', cmap='inferno')
    # pl.colorbar(hb)
    # pl.xlabel('Postive Experience (PE)')
    # pl.ylabel('Postive Distance (PD)')
    # pl.title('Student T-Mixtures for Emotion regulation')
    # X = np.vstack([X.image_data_masked, Y.image_data_masked]).transpose()
    for j in range(1,21,1):
        mean1 = (1, 2)
        cov1 = [[1, 0.8], [0.8, 1]]
        X1 = multivariate_t_rvs(mean1, cov1,df=10, n=10000)
        mean2 = (j, 2)
        cov2 = [[1, 0.3], [0.3, 1]]
        X2 = multivariate_t_rvs(mean2, cov2,df=20,n=10000)
        X = np.vstack([X1,X2])

        # pl.scatter(X1[:,0],X1[:,1])
        # pl.scatter(X2[:,0],X2[:,1],color='green')
        # pl.title("T-Distribution Mixture model (diff = 1)")

        data = DataSet()
        data.fromArray(X)

        #Initialize Mean, Weights, Covariance and dof of the Components
        means_ = KMeans(n_clusters=n_components).fit(X).cluster_centers_
        weights_ = np.tile(1.0 / n_components, n_components)

        min_covar=1e-3
        tied_cv = np.cov(X.T) + min_covar * np.eye(X.shape[1])
        cv = np.tile(tied_cv, (n_components, 1, 1))
        df = 10.

        #Initialize the mixture
        comp = []
        for i in range(n_components):
        	comp.append(TD(n_dim,means_[i],cv[i],df))

        m = mixture(n_components,weights_, comp)

        #Run the EM step
        m.EM(data,max_iter,tol)
        print m.loglikelihood
        output[n] = m.loglikelihood
        output_arr.append(m.loglikelihood)
        anchor = ""
        density = []
        for i in range(n_components):
            mean = m.components[i][0].mu
            cv = m.components[i][0].sigma
            dof = m.components[i][0].df
            density.append(m.components[i][0].pdf(data))
            label = str(i+1)
            # pl.scatter(mean[0],mean[1],color='red')
            # pl.annotate(label,xy=(mean[0], mean[1]), xytext=(-20, 20),textcoords='offset points', ha='right', va='bottom',bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            # plot_cov_ellipse(cv,mean)
            anchor += "Comp: "+str(i+1)+" pi: "+str(m.pi[n_components-i-1])+"\n"
            print mean,cv,dof, m.pi[n_components-i-1]

    # p_values = []
    # chi_stat = []
    # for i in range(n_components):
    #     temp_p = []
    #     temp_chi  = []
    #     for j in range(len(data)):
    #         l0 = density[i][j]
    #         l1 = 0
    #         for k in range(n_components):
    #             if k != i:
    #                 l1 += density[k][j]
    #         chi_square = 2 * float(l0-l1)
    #         temp_chi.append(chi_square)
    #         p = stats.distributions.chi2.sf(chi_square, n_components-1)
    #         temp_p.append(p)
    #     p_values.append(temp_p)
    #     chi_stat.append(temp_chi)



# output_arr = np.asarray(output_arr)
# ax = pl.gca()
# ax.add_artist(AnchoredText(anchor, loc=2))


# print "Minimum",np.max(output_arr), np.where(output_arr==np.max(output_arr))[0]
print output_arr
x = range(1,21)
y = [ -1 * i for i in output_arr]
pl.xticks(x)
pl.title('LogLikelihood vs Distance between components (n=2)')
pl.xlabel('diff')
pl.ylabel('Negative loglikelihood')
pl.xlim(1,22)
pl.plot(x,y)
pl.show()