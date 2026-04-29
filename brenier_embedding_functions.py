import os
import numpy as np
import torch
import ot
from torch.distributions import MultivariateNormal
from collections import Counter



def center_ot_dual(alpha0, beta0, a=None, b=None):
    r"""Center dual OT potentials w.r.t. their weights

    The main idea of this function is to find unique dual potentials
    that ensure some kind of centering/fairness. The main idea is to find dual potentials that lead to the same final objective value for both source and targets (see below for more details). It will help having
    stability when multiple calling of the OT solver with small changes.

    Basically we add another constraint to the potential that will not
    change the objective value but will ensure unicity. The constraint
    is the following:

    .. math::
        \alpha^T \mathbf{a} = \beta^T \mathbf{b}

    in addition to the OT problem constraints.

    since :math:`\sum_i a_i=\sum_j b_j` this can be solved by adding/removing
    a constant from both  :math:`\alpha_0` and :math:`\beta_0`.

    .. math::
        c &= \frac{\beta_0^T \mathbf{b} - \alpha_0^T \mathbf{a}}{\mathbf{1}^T \mathbf{b} + \mathbf{1}^T \mathbf{a}}

        \alpha &= \alpha_0 + c

        \beta &= \beta_0 + c

    Parameters
    ----------
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    a : (ns,) numpy.ndarray, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) numpy.ndarray, float64
        Target histogram (uniform weight if empty list)

    Returns
    -------
    alpha : (ns,) numpy.ndarray, float64
        Source centered dual potential
    beta : (nt,) numpy.ndarray, float64
        Target centered dual potential

    """
    # if no weights are provided, use uniform
    if a is None:
        a = np.ones(alpha0.shape[0]) / alpha0.shape[0]
    if b is None:
        b = np.ones(beta0.shape[0]) / beta0.shape[0]

    # compute constant that balances the weighted sums of the duals
    c = (b.dot(beta0) - a.dot(alpha0)) / (a.sum() + b.sum())

    # update duals
    alpha = alpha0 + c
    beta = beta0 - c

    return alpha, beta


def brenier_potential(source_points,source_masses, target_points,target_masses, method,eps_reg):
    #Remark: ot.emd and ot.bregman.sinkhorn_stabilized normalize the potentials so that E[f]=E[g]=1/2*cost
    if type(target_points)!=np.ndarray:
        target_points=target_points.numpy()
    if type(source_points)!=np.ndarray:
        source_points=source_points.numpy()
    if source_masses==None:
        n1 = np.shape(source_points)[0]
        source_masses=np.ones(n1)/n1
    if target_masses==None:
        n2 = np.shape(target_points)[0] 
        target_masses=np.ones(n2)/n2 
    M = ot.dist(source_points, target_points,metric='sqeuclidean')
    M = M.astype('float64')
    max_value=M.max()
    M /= max_value
    if method == 'emd':
        OTplan,log_dict = ot.emd(source_masses, target_masses, M, numItermax = 1e7,log=True)
        u=log_dict['u']
    elif method == 'entropic':
        if eps_reg==None:
            eps_reg=5*1e-3
        OTplan,log_dict = ot.sinkhorn(source_masses, target_masses, M, reg = eps_reg,log=True)
       # OTplan,log_dict = ot.sinkhorn(source_masses, target_masses, M, reg = eps_reg,log=True,method='sinkhorn_stabilized')
        u=log_dict['u']
        v=log_dict['v']
        tiny=1e-20
        u=eps_reg*np.log(np.maximum(u, tiny))
        v=eps_reg*np.log(np.maximum(v, tiny))
        u=u-np.mean(u)
    return torch.tensor(u)*max_value


def wass_map(source_points,source_masses, target_points,target_masses, method):
    if type(source_points)!=np.ndarray:
        source_points = np.asarray(source_points, dtype=np.float64)

    if type(target_points)!=np.ndarray:
        target_points = np.asarray(target_points, dtype=np.float64)
    source_points = np.nan_to_num(source_points, nan=0.0, posinf=0.0, neginf=0.0)
    target_points = np.nan_to_num(target_points, nan=0.0, posinf=0.0, neginf=0.0)
    p = source_points.shape[1]

    M = ot.dist(source_points, target_points)
    M = M.astype('float64')
    M /= M.max()
    if source_masses==None:
        source_masses=np.ones(len(source_points))/len(source_points)
    if target_masses==None:
        target_masses=np.ones(len(target_points))/len(target_points)

    if method == 'emd':
        OTplan = ot.emd(source_masses, target_masses, M, numItermax = 1e7)
    elif method == 'entropic':
        OTplan = ot.bregman.sinkhorn_stabilized(source_masses, target_masses, M, reg = 5*1e-3)
    # initialization
    OTmap = np.empty((0, p))
    for i in range(len(source_points)):
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])
        # conditional expectation
        OTmap = np.vstack([OTmap, (np.transpose(target_points) @ OTplan[i,:])])
    OTmap = np.array(OTmap).astype('float32')
    return torch.tensor(OTmap)

def wass_map_1D(source_points, source_masses, target_points, target_masses,
                *, device=None, dtype=torch.float32):
    """
    1D optimal transport barycentric map.
    
    Returns: torch.Tensor of shape (n, 1) - mapped locations for each source point.
    """
    # Convert to 1D arrays
    xs = np.asarray(source_points, dtype=np.float64).reshape(-1)
    xt = np.asarray(target_points, dtype=np.float64).reshape(-1)
    
    n, m = len(xs), len(xt)
    
    # Handle masses (default to uniform if None)
    a = np.full(n, 1.0/n) if source_masses is None else np.asarray(source_masses).reshape(-1)
    b = np.full(m, 1.0/m) if target_masses is None else np.asarray(target_masses).reshape(-1)
    
    # Normalize masses
    a = a / a.sum()
    b = b / b.sum()
    
    # Compute OT plan
    G = ot.emd_1d(xs, xt, a, b, metric="sqeuclidean")
    
    # Barycentric projection: T(x_i) = weighted average of target points
    T = (G @ xt) / G.sum(axis=1)
    
    return torch.as_tensor(T.reshape(-1, 1), dtype=dtype, device=device)



def generate_brenier_tensors(base_measure,target_measure_list):
    #base_measure: measure
    #data: list of measures
    list_of_potentials=[]
    for i in range(len(target_measure_list)):
        if i%100==0:
            print(i)
        potential=brenier_potential(base_measure,target_measure_list[i],'emd')
        list_of_potentials.append(torch.tensor(potential,dtype=torch.float32))
    potential_tensor=torch.stack(list_of_potentials)
    return potential_tensor

'''
def compute_brenier_tensor_from_list_gaussian_base(list_of_arrays,supp_size,variance_scaling, save_dir):
    #Put None as save_dir if you don't want to save anything
    if save_dir!=None:
        os.makedirs(save_dir,exist_ok=True)
    variances = []
    support_sizes = []

    for i in range(len(list_of_arrays)):
        cloud = torch.tensor(list_of_arrays[i], dtype=torch.float32)  
        cloud_var = cloud.var(dim=0, unbiased=False)
        variances.append(cloud_var)
        support_sizes.append(cloud.shape[0])

    avg_variance = torch.stack(variances).mean(dim=0)  # shape (3,)

    # Sample base measure from Gaussian
    gaussian = MultivariateNormal(
        loc=torch.zeros_like(avg_variance),
        covariance_matrix=torch.diag(variance_scaling*avg_variance)
    )
    
    gaussian_sample = gaussian.sample((supp_size,))  # shape (avg_support_size, 3)
    gaussian_sample = gaussian_sample.numpy()
    # Save base_measure
    if save_dir!=None:
        os.makedirs(os.path.join(save_dir, "base_measure"), exist_ok=True)
        torch.save(torch.tensor(gaussian_sample), os.path.join(save_dir, "base_measure", "base_measure.pt"))
    # Resample normalized point clouds to match the base measure size if needed
    measure_list = []
    for array in list_of_arrays:
        # Convert cloud to tuples so we can count duplicates
        support_tuple = [tuple(p) for p in array]
        # Count occurrences of each unique vector
        counts = Counter(support_tuple)
        unique_points = np.array(list(counts.keys()))
        frequencies = np.array(list(counts.values()), dtype=float)
        masses = frequencies / frequencies.sum()
        # Append measure for this cloud
        measure_list.append(measure(unique_points, masses))
    gaussian_measure = measure(gaussian_sample, np.ones(supp_size)/supp_size)
    # Generate mapping tensor
    potential_tensor = generate_brenier_tensors(gaussian_measure, measure_list)
    if save_dir!=None:
        torch.save(torch.tensor(potential_tensor),os.path.join(save_dir,"brenier_potential_tensor.pt"))
    return potential_tensor, gaussian_sample
'''