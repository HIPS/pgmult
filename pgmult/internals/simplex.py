"""
From Matt's dirichlet-truncated-multinomial repo.
"""


import numpy as np

def mesh(N,edges=True):
    if not edges:
        N = N-1

    tups = []
    for count, x1 in enumerate(np.linspace(0.,1.,num=N,endpoint=edges)):
        for x2 in np.linspace(0.,1.-x1,num=N-count,endpoint=edges):
            x3 = 1-x1-x2
            tups.append((x1,x2,x3))
    tups = np.array(tups)
    if not edges:
        tups = tups[np.logical_not((tups == 0).any(axis=1))]
    return tups

def meshnd(D, N):
    tups = []

    def _compute_points_helper(x, C, stick, d):
        # Recursively compute the set of points for partial vector x
        # with "count" points in the (d-1) dimension

        if d == D:
            tups.append(x)
        else:
            for c, xd in enumerate(np.linspace(0., stick, num=C)):
                _compute_points_helper(x + (xd,), C-c, stick-xd, d+1)

    # Start
    _compute_points_helper((), N, 1.0, 1)

    # Convert tups to array and complete the last dimension
    tups = np.array(tups)
    tups = np.hstack((tups, (1.0 - tups.sum(axis=1))[:,None]))

    return tups

def nonlinear_meshnd(D, N=10):
    """
    :param D:  1+dimensionality of the simplex
    :param N:  number of points per dimension
    :param edges:
    :return:
    """
    # Compute the points
    inds = np.array(np.unravel_index(np.arange(N**D), (N,)*D))
    inds = inds.T
    inds = inds.astype(np.float)

    # The point is simply the normalized index
    # Firs we remove the point (0,0,...,0)
    inds = inds[1:]
    pts = inds / inds.sum(1)[:,None]

    # Remove redundant points
    pts = np.unique(pts)

    assert np.allclose(pts.sum(1), 1.0)
    assert np.amin(pts) >= 0
    assert np.amax(pts) <= 1

    return pts

def proj_to_2D(points):
    # this special case is useful; it's slightly numerically nicer and so it
    # plays better with delaunay triangulation algorithms, which I use for 2D
    # plotting
    return np.dot(points,np.array([[0,1],[1,-0.5],[-1,-0.5]]))

def _get_projector(n):
    foo = np.ones((n,n-1))
    foo[np.arange(n-1),np.arange(n-1)] = -(n-1)
    Q,R = np.linalg.qr(foo)
    return Q

def proj_vec(v):
    v = np.array(v,ndmin=2)
    Q = _get_projector(v.shape[1])
    return np.dot(v,Q)

def proj_matrix(mat):
    Q = _get_projector(mat.shape[1])
    return Q.T.dot(mat).dot(Q)

def test():
    from matplotlib import pyplot as plt

    mesh3D = mesh(25,edges=True)
    mesh2D = proj_to_2D(mesh3D)
    plt.figure(); plt.plot(mesh2D[:,0],mesh2D[:,1],'bo',label='incl. edges')
    mesh3D_noedges = mesh(25,edges=False)
    mesh2D_noedges = proj_to_2D(mesh3D_noedges)
    plt.plot(mesh2D_noedges[:,0],mesh2D_noedges[:,1],'rx',label='excl. edges')
    plt.legend()

    from mpl_toolkits.mplot3d import Axes3D # this import is needed, really!
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(mesh3D[:,0],mesh3D[:,1],mesh3D[:,2],c='b')

    plt.show()
