

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

max_comp = 9
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
    x_image_pname = 'PE_m.nii.gz'
    x_mask_pname =  'mask_wm.nii.gz'  #self masked
    y_image_pname = 'PD_m.nii.gz'
    y_mask_pname =  'mask_wm.nii.gz'
    X = CCAW.BaseData(x_image_pname, x_mask_pname)
    Y = CCAW.BaseData(y_image_pname, y_mask_pname)
    # hb = pl.hexbin(X.image_data_masked, Y.image_data_masked, gridsize=50, bins='log', cmap='inferno')
    # pl.colorbar(hb)
    pl.xlabel('Postive Experience (PE)')
    pl.ylabel('Postive Distance (PD)')
    # pl.title('Student T-Mixtures for Emotion regulation')
    X = np.vstack([X.image_data_masked, Y.image_data_masked]).transpose()


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

    list_index = []
    for i in range(len(data)):
        data_pont = X[i]
        temp = []
        for j in range(n_components):
            temp.append(density[j][i])
        temp = np.asarray(temp)
        list_index.append(np.where(temp==np.max(temp))[0])

    from random import randint
    colors = ['blue','green','red','sienna','violet','orange','turquoise','khaki','gold']
    pl.title("Scatter plot of Components (Neyman pearson lemma)")
    list_index = np.asarray(list_index)
    component_data =[]
    for i in range(n_components):
        model = nibabel.load('mask_wm.nii.gz')
        model_data = model.get_data().flatten()
        model_index = np.where(model_data!=0.)[0]
        temp = np.zeros(len(data))
        temp_index = np.where(list_index==i)[0]
        color_plt = []

        for index in temp_index:
            color_plt.append(X[index])
        color_plt = np.asarray(color_plt)
        pl.scatter(color_plt[:,0],color_plt[:,1],color=colors[i])
        temp[temp_index] = 1.
        image_data =  np.zeros(model_data.shape)
        image_data[model_index] = temp
        temp_nifti = nibabel.Nifti1Image(image_data.reshape(model.shape), affine=model.get_affine())
        print 'Saving: ',str(i)+'.nii.gz'
        nibabel.save(temp_nifti, str(i)+'.nii.gz')

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



output_arr = np.asarray(output_arr)
# ax = pl.gca()
# ax.add_artist(AnchoredText(anchor, loc=2))


# print "Minimum",np.max(output_arr), np.where(output_arr==np.max(output_arr))[0]
print output_arr
# x = output.keys()
# y = [ -1 * output[i] for i in output.keys()]
# pl.xticks(x)
# pl.title('Scree plot for mixtures of Student t-distribution')
# pl.xlabel('Number of Components')
# pl.ylabel('Negative loglikelihood')
# pl.xlim(1,max_comp+1)
# pl.plot(x,y)
pl.show()