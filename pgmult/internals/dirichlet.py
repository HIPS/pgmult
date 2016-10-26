"""
From Matt's dirichlet-truncated-multinomial repo.
"""



import numpy as np
na = np.newaxis
import scipy.special

from .simplex import proj_to_2D, mesh

def _dirichlet_support_check(x,alpha):
    x = np.array(x,ndmin=2)
    alpha = np.array(alpha,ndmin=1)
    assert alpha.ndim == 1
    if len(alpha) == 1:
        alpha = alpha * np.ones(x.shape[1])
    else:
        assert x.shape[1] == len(alpha)

    retvals = np.zeros(x.shape[0])
    goodindices = np.logical_and((x >= 0).all(1),np.abs(x.sum(1)-1.) < 1e-6)
    badindices = np.logical_not(goodindices)
    retvals[badindices] = -np.inf

    return retvals, goodindices, badindices, x, alpha

def log_censored_dirichlet_density(x,alpha,data=None):
    '''
    UNNORMALIZED symmetric censored dirichlet density

    x is an NxP set of query points on which to evaluate the P-dimensional density

    data is an optional PxP matrix where the ijth entry is the number of times
    face j came up during the i-censored rounds. therefore np.all(np.diag(data)==0)
    '''
    retvals, goodindices, badindices, x, alpha = _dirichlet_support_check(x,alpha)
    if goodindices.sum() > 0:
        x = x[goodindices]
        logvals = ((alpha-1) * np.log(x)).sum(axis=1)
        if data is not None:
            assert (np.diag(data) == 0).all(), 'censored!'
            for idx, row in enumerate(data):
                ins = row*(np.log(x) - np.log(1.-x[:,idx])[:,na])
                logvals += np.where(np.isnan(ins),0.,ins).sum(1)
        retvals[goodindices] = logvals
    return retvals

def log_dirichlet_density(x,alpha,data=None):
    retvals, goodindices, badindices, x, alpha = _dirichlet_support_check(x,alpha)

    if goodindices.sum() > 0:
        x = x[goodindices]
        logvals = ((alpha-1) * np.log(x)).sum(axis=1) - scipy.special.gammaln(alpha).sum() + scipy.special.gammaln(alpha.sum())
        if data is not None:
            assert data.ndim == 1 and data.shape[0] == x.shape[1]
            ins = data*np.log(x)
            logvals += np.where(np.isnan(ins),0.,ins).sum(axis=1) # 0^0 := 1 => 0*log(0) = 0.
        retvals[goodindices] = logvals
    return retvals

def test_pcolor_heatmap():
    # import matplotlib.tri as tri
    from matplotlib import pyplot as plt

    mesh3D = mesh(100,edges=True)
    mesh2D = proj_to_2D(mesh3D)
    # triangulation = tri.Triangulation(mesh2D) # this is called in tripcolor

    data = np.zeros((3,3))
    data[0,1] += 1

    vals = np.exp(log_dirichlet_density(mesh3D,2.,data=data.sum(0)))
    temp = log_censored_dirichlet_density(mesh3D,2.,data=data)
    censored_vals = np.exp(temp - temp.max())

    plt.figure()
    plt.tripcolor(mesh2D[:,0],mesh2D[:,1],vals)
    plt.title('uncensored')

    plt.figure()
    plt.tripcolor(mesh2D[:,0],mesh2D[:,1],censored_vals)
    plt.title('censored')

def test_imshow_heatmap():
    from scipy.interpolate import griddata
    from matplotlib import pyplot as plt

    mesh3D = mesh(200)
    mesh2D = proj_to_2D(mesh3D)

    data = np.zeros((3,3))
    data[0,1] += 2

    vals = np.exp(log_dirichlet_density(mesh3D,2.,data=data.sum(0)))
    temp = log_censored_dirichlet_density(mesh3D,2.,data=data)
    censored_vals = np.exp(temp - temp.max())

    xi = np.linspace(-1,1,1000)
    yi = np.linspace(-0.5,1,1000)

    plt.figure()
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),vals,(xi[None,:],yi[:,None]),method='cubic'))
    plt.axis('off')
    plt.title('uncensored likelihood')

    plt.figure()
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),censored_vals,(xi[None,:],yi[:,None]),method='cubic'))
    plt.axis('off')
    plt.title('censored likelihood')


def test_contour_heatmap():
    from scipy.interpolate import griddata
    from matplotlib import pyplot as plt

    mesh3D = mesh(200)
    mesh2D = proj_to_2D(mesh3D)

    data = np.zeros((3,3))
    data[0,1] += 2

    vals = np.exp(log_dirichlet_density(mesh3D,2.,data=data.sum(0)))
    temp = log_censored_dirichlet_density(mesh3D,2.,data=data)
    censored_vals = np.exp(temp - temp.max())

    xi = np.linspace(-1,1,1000)
    yi = np.linspace(-0.5,1,1000)

    plt.figure()
    plt.contour(griddata((mesh2D[:,0],mesh2D[:,1]),vals,(xi[None,:],yi[:,None]), method='cubic'))
    plt.axis('off')
    plt.title('uncensored likelihood')

    plt.figure()
    plt.contour(griddata((mesh2D[:,0],mesh2D[:,1]),censored_vals,(xi[None,:],yi[:,None]),method='cubic'))
    plt.axis('off')
    plt.title('censored likelihood')