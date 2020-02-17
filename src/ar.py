import numpy as np
import numpy.linalg as linalg

def state_space(raw_data, q):
    """
    Performs the state-space projection of the original data using principal
    component analysis (eigen-decomposition).

    Parameters
    ----------
    raw_data : array, shape (N, M)
        Row-vector data points with M features.
    q : integer
        Number of principal components to keep.

    Returns
    -------
    X : array, shape (q, M)
        State-space projection of the original data.
    C : array, shape (N, q) the PCA matrix (useful for returning to the data space)
        Projection matrix.
    """
    if q <= 0:
      raise Exception('Parameter "q" restricted to positive integer values.')

    # Perform the SVD on the data.
    # For full documentation on this aspect, see page 15 of Midori Hyndman's
    # master's thesis on Autoregressive modeling.
    #
    # Y = U * S * Vt,
    #
    # Y = C * X,
    #
    # So:
    # C = first q columns of U
    # S_hat = first q singular values of S
    # Vt_hat = first q rows of Vt
    #
    # X = S_hat * Vt_hat
    #
    # For the full documentation of SVD, see:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
    U, S, Vt = linalg.svd(raw_data, full_matrices = False)
    C = U[:, :q]
    Sh = np.diag(S)[:q, :q]
    Vth = Vt[:q, :]
    X = np.dot(Sh, Vth)
    return [X, C]

def appearance_space(state_data, C):
    """
    Converts data projected into the state space back into the appearance space
    according to the projection matrix C. Effectively, this undoes the operations
    of "state_space()":

    X, C = state_space(original_data)
    original_data = appearance_space(X, C)

    Parameters
    ----------
    state_data : array, shape (q, M)
        The projected data, or the output "X" from "state_space".
    C : array, shape (N, q)
        The projection matrix, or the output "C" from "state_space".

    Returns
    -------
    X : array, shape (N, M)
        The original form of the data, or the input to "state_space".
    """
    return np.dot(C, state_data)

def train(X, order = 2):
    """
    Estimates the transition matrices A (and eventually the error parameters as
    well) for this AR model, given the order of the markov process.

    (in this notation, the parameter to this method "order" has the same value as "q")

    Parameters
    ----------
    X : array, shape (q, M) or (M,)
        Matrix of column vectors of the data (either original or state-space).
    order : integer
        Positive, non-zero integer order value for the order of the Markov process.

    Returns
    -------
    A : array, shape (q, q)
        Transition coefficients for the system
    """
    if order <= 0:
        raise Exception('Parameter "order" restricted to positive integer values')
    W = None

    # A particular special case first.
    if len(X.shape) == 1:
        Xtemp = np.zeros(shape = (1, np.size(X)))
        Xtemp[0, :] = X
        X = Xtemp

    # What happens in this loop is so obscenely complicated that I'm pretty
    # sure I couldn't replicate if I had to, much less explain it. Nevertheless,
    # this loop allows for the calculation of n-th order transition matrices
    # of a high-dimensional system.
    #
    # I know this could be done much more simply with some np.reshape() voodoo
    # magic, but for the time being I'm entirely too lazy to do so. Plus, this
    # works. Which is good.
    for i in range(1, order + 1):
        Xt = X[:, order - i: -i]
        if W is None:
            W = np.zeros((np.size(Xt, axis = 0) * order, np.size(Xt, axis = 1)))
        W[(i - 1) * np.size(Xt, axis = 0):((i - 1) * np.size(Xt, axis = 0)) + np.size(Xt, axis = 0), ...] = Xt
    Xt = X[:, order:]
    A = np.dot(Xt, linalg.pinv(W))
    
    # The data structure "A" is actually all the transition matrices appended
    # horizontally into a single NumPy array. We need to extract them.
    matrices = []
    for i in range(0, order):
        matrices.append(A[..., i * np.size(A, axis = 0):(i * np.size(A, axis = 0)) + np.size(A, axis = 0)])
    return matrices

def test(X, A, guided = True):
    """
    Parameters
    ----------
    X : array, shape (N, M)
        The original data (lots of column vectors).
    A : array, shape (q, q)
        List of transition matrices or coefficients (output from estimate_parameters).
    guided : boolean
        Indicates whether or not this is a "guided" reconstruction. If True 
        (default), each "order" number of points is used to predict a new point, 
        but the predicted point is not in turn used to predict the next point.
        If False, predicted points are used to predict new points.

    Returns
    -------
    Xrecon : array, shape (N, M)
        The reconstructed data. Same dimensions as X. Hopefully similar
        quantities as well.
    """
    # The order of the markov process is, not all coincidentally, the
    # number of transition matrices we have in this list.
    order = np.size(A, axis = 0)

    # This is somewhat tricky. For abitrary order, we need to 
    # come up with an expression for:
    #
    # Xrecon = SUM_OF_PREVIOUS_TERMS
    #
    # where SUM_OF_PREVIOUS_TERMS is constructed in a loop over "order" 
    # previous elements in the data, multiplying each element by the 
    # corresponding transition matrix/coefficient. Then that sum needs to be
    # but a single element in a larger array that has a correspondence to
    # the original X.
    Xrecon = np.zeros(X.shape)
    Xrecon[:, :order] = X[:, :order]
    for i in range(order, np.size(X, axis = 1)):
        for j in range(1, order + 1):

            # The second argument to np.dot() is a ternary statement, conditioning
            # on the "guided" boolean passed into this method: do we use the actual
            # data in estimating the next point, or previously-esimated data?
            Xrecon[:, i] += np.dot(A[j - 1], X[:, i - j] if guided else Xrecon[:, i - j])
    return Xrecon - np.mean(Xrecon, axis = 0)

def error(Y, X):
    """
    Calculates mean squared error (MSE) of the reconstructed data (X) relative
    to the original data (Y). In theory, this number should decrease as the
    order of the markov process increases and/or the number of components
    involved in the original projection (if used) increases.

    Parameters
    ----------
    Y : array, shape (N, M)
        The original data.
    X : array, shape (N, M)
        The reconstructed data (output from "test").

    Returns
    -------
    MSE : array, shape (N, M)
        Same dimensions as X and Y, where each element is the MSE as if the
        corresponding element in the reconstructed data Y operated as the estimator 
        for the corresponding element in the original data X.
    """
    return (Y - X) ** 2
