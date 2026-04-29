import ot
import numpy as np
import torch

class measure:
    def __init__(self,points,masses):
        self.points=points
        self.masses=masses

class embedded_data:
    def __init__(self,mapping,label,data,base,true_coefficients):
        self.mapping = mapping
        self.label=label
        self.data=data
        self.base=base
        self.true_coefficients=true_coefficients

class scored_data:
    def __init__(self, embedded_data,score,regressed_coefficients):
        self.embedded_data = embedded_data
        self.score=score
        self.regressed_coefficients=regressed_coefficients


def divide_by_label(dataset):
    label_dict = {}
    for item in dataset:
        label = item.label
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(item)
    return label_dict

def wass_map(source, target, method):
    n1 = source.points
    n2 = target.points  
    p = source.points.shape[1]
    M = ot.dist(source.points, target.points)
    M = M.astype('float64')
    M /= M.max()
    if method == 'emd':
        OTplan = ot.emd(source.masses, target.masses, M, numItermax = 1e7)
    elif method == 'entropic':
        OTplan = ot.bregman.sinkhorn_stabilized(source.masses, target.masses, M, reg = 5*1e-3)
    # initialization
    OTmap = np.empty((0, p))
    for i in range(len(n1)):
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])
        # conditional expectation
        OTmap = np.vstack([OTmap, (np.transpose(n2) @ OTplan[i,:])])
    OTmap = np.array(OTmap).astype('float32')
    return OTmap


def image_to_empirical(image:np.array):
    '''
    image_to_empirical - Converts an image into an empirical measure which tracks support and mass    
    :param image: (n x m) np array representing an image
    :return:      (l x 2) np array of the support location and (l) np array of mass
    '''
    [height, width] = image.shape
    # for normalizing the height to be between 0 and 1
    # handles the edge case of height or width being 1 pixel
    nheight = max(height - 1, 1) 
    nwidth = max(width - 1, 1)
    support = []
    mass = []
    for i in range(height):
        for j in range(width):
            if image[i,j] == 0:
                continue
            support += [[i /  nheight, j / nwidth]]
            mass += [image[i,j]]
    return measure(np.array(support), np.array(mass)/np.sum(mass))


#SAMPLING

def sample_from_simplex(n):
    """
    Uniformly sample a point from the n-simplex.
    
    :param n: Dimension of the simplex (n-simplex has n+1 vertices).
    :return: A point in the n-simplex as a 1D NumPy array of size (n+1,).
    """
    # Generate n random numbers from the uniform distribution (0, 1)
    random_numbers = np.random.rand(n)
    
    # Sort the random numbers
    sorted_numbers = np.sort(random_numbers)
    
    # Add 0 at the beginning and 1 at the end to form the "breaks"
    breaks = np.concatenate(([0], sorted_numbers, [1]))
    
    # Compute the differences between consecutive breaks
    simplex_point = np.diff(breaks)
    
    return simplex_point

def generate_grid_measure(n_points):
    #Will generate int(sqrt{n_points})**2 points on a grid with uniform masses
    n_side = int(np.sqrt(n_points))
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xv, yv = np.meshgrid(x, y)
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    masses=np.dot(np.ones(len(points)),1/len(points))
    return measure(points,masses)

#sample_displacements outputs a torch.tensor since model is a NN
#build_QP and solve_QP operates on arrays.
#conversion between the two is handled in solve_for_coefficients

#inputs: torch.nn, torch.tensor
def sample_displacements(model, base_points):
    images = torch.stack([model(point) for point in base_points])  # Properly stack tensors
    return base_points - images

#input: np.array and int
def build_QP(map_array,base_supp_size):
    map_array=np.array(map_array)
    m=len(map_array)
    Q=np.zeros((m,m))
    normalization=1/base_supp_size
    for i in np.arange(m):
        for j in np.arange(m):
            Q[i,j]=np.dot(normalization,np.sum(np.sum(map_array[i]*map_array[j],axis=1)))
    return Q


#input: np.array
def solve_QP(matrix):
    matrix=np.array(matrix)
    x=cp.Variable(len(matrix))
    objective=cp.Minimize(cp.quad_form(x,matrix))
    constraints= [x>=0,cp.sum(x)==1]
    problem=cp.Problem(objective,constraints)
    problem.solve()
    optimal_x=x.value
    return optimal_x, problem.value

def build_QP_v2(displacements):
    #displacements: (batch_size,no_heads,supp_size,dim)
    batch_size=displacements.shape[0]
    no_heads=displacements.shape[1]
    supp_size=displacements.shape[2]
    Q=torch.zeros(batch_size,no_heads,no_heads)
    normalization=1/supp_size
    for i in np.arange(no_heads):
        for j in np.arange(no_heads):
            prod=torch.mean(torch.sum(displacements[:,i,:,:]*displacements[:,j,:,:],dim=2),dim=1) #prod: batch_size
            Q[:,i,j]=normalization*prod
    return Q


def multi_solve_for_coefficients(model,base_points,mapping,QP_reg):
    #base_points: (supp_size,dim)
    #mapping: (supp_size,dim)
    with torch.no_grad():
        output=model(base_points) #output: (supp_size,no_heads,dim)
    no_heads=output.shape[1]
    supp_size=base_points.shape[0]
    map_array=[]
    for i in range(no_heads):
        tensor_map=output[:,i,:]-mapping #tensor_map: (supp_size,dim)
        map_array.append(tensor_map.cpu().detach().numpy())
    #build and solve QP
    QP_mat=build_QP(map_array,supp_size)+QP_reg*np.eye(len(map_array))
    coefficients,min_value=solve_QP(QP_mat)
    return coefficients,min_value

def multi_solve_for_coefficients(model,base_points,mapping,QP_reg,):
    #base_points: (supp_size,dim)
    #mapping: (supp_size,dim)
    with torch.no_grad():
        output=model(base_points) #output: (supp_size,no_heads,dim)
    no_heads=output.shape[1]
    supp_size=base_points.shape[0]
    map_array=[]
    for i in range(no_heads):
        tensor_map=output[:,i,:]-mapping #tensor_map: (supp_size,dim)
        map_array.append(tensor_map.cpu().detach().numpy())
    #build and solve QP
    QP_mat=build_QP(map_array,supp_size)+QP_reg*np.eye(len(map_array))
    coefficients,min_value=solve_QP(QP_mat)
    return coefficients,min_value


class reg_schedule:
    def __init__(self, base_value, scale_factor,min_i):
        # Example parameters to define the schedule
        self.base_value = base_value
        self.scale_factor = scale_factor
        self.min_i=min_i

    def __call__(self, i):
        if i<=self.min_i:
            return self.base_value
        else:
            return self.base_value /((i-self.min_i)*self.scale_factor)



#def evaluate_coefficients(data_list,model):
#    scored_data_list=[]
#    for data in data_list:
#        base_points=torch.tensor(data.base.points,dtype=torch.float32)
#        mapping=torch.tensor(data.mapping,dtype=torch.float32)
#        coeff,min_value=multi_solve_for_coefficients(model,base_points,mapping,0)
#        scored_data_list.append(scored_data(data,min_value,coeff))
#    return scored_data_list

