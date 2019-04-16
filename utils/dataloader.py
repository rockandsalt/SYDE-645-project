import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import h5py
from numba import jit, prange
from sklearn.model_selection import StratifiedShuffleSplit

def sampler(sample_size):
    """function sample n stl file from each folder
    
    Arguments:
        sample_size {int} -- number of sample
    
    Returns:
        list -- list of stl file path
    """

    path = os.path.join('.','dataset','stl')
    tree = os.walk(path)

    list_stl_file = []

    for root, dirs, files in tree:
        if(len(files) != 0):
            join_root = lambda x, r = root: os.path.join(r,x)
            if(sample_size == -1):
                sample = files
            else:
                sample = np.random.choice(files,sample_size)
            ap = [os.path.join(root,x) for x in sample]
            list_stl_file.append(ap)
        
    return list_stl_file

def load_file(path):
    """from path load stl and voxelize it
    
    Arguments:
        path {string} -- file path
    
    Returns:
        vtkImageData -- vtk voxel data type
    """

    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(path)
    stl_reader.Update()

    bounds = stl_reader.GetOutput().GetBounds()

    vox_modeller = vtk.vtkVoxelModeller()
    vox_modeller.SetSampleDimensions(64,64,64)
    vox_modeller.SetModelBounds(bounds)
    vox_modeller.SetScalarTypeToInt()
    vox_modeller.SetMaximumDistance(.1)

    vox_modeller.SetInputConnection(stl_reader.GetOutputPort())
    vox_modeller.Update()

    return vox_modeller.GetOutput()

def convert_image_to_numpy(vtk_image):
    """takes vtkImageData convert it to numpy
    
    Arguments:
        vtk_image {vtkImageData} -- vtk voxel data type
    
    Returns:
        np.array -- numpy array
    """

    dim = vtk_image.GetDimensions()
    sc = vtk_image.GetPointData().GetScalars()
    arr = vtk_to_numpy(sc)
    return arr.reshape(dim)

def convert_all_data(output_path):
    svc_path = sampler(-1)

    i,j = np.shape(svc_path)

    output_data_p = os.path.join(output_path,'data.hdf5')
    hf = h5py.File(output_data_p, 'w')

    data_set = hf.create_dataset("data", (i*j,64**3))
    data_label = hf.create_dataset("data_label", (i*j,),dtype='i8')

    for label_id in range(i):
        for path_id in range(j):
            path = svc_path[label_id][path_id]
            vox = load_file(path)
            arr = convert_image_to_numpy(vox)
            data_set[label_id*j+path_id] = arr.reshape(64**3)
            data_label[label_id*j+path_id] = label_id

    hf.close()

### helper function for multithreading
def opti_func(train_i,test_i,crt_clf,X,Y):
    filt,clf = crt_clf()
    
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = Y[train_i], Y[test_i]
    
    X_train = filt.fit_transform(X_train, y_train)
    X_test = filt.transform(X_test)
    clf.fit(X_train, y_train)
    
    return clf.score(X_test,y_test)*100

def load_data(path_str):
    dat = h5py.File(path_str, 'r')

    X = np.array(dat.get('data'))
    Y = np.array(dat.get('data_label'))
    dat.close()

    return (X,Y)

def split_data(input_path,output_path,name):

    dat = h5py.File(input_path, 'r')

    X = np.array(dat.get('data'))
    Y = np.array(dat.get('data_label'))

    length = X.shape[0]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

    for s_data in sss.split(X,Y):
        for i in range(2):
            dat_i = h5py.File(os.path.join(output_path,'{}_{}.hdf5'.format(name,i)), 'w')
            data_i_set = dat_i.create_dataset("data", (length/2,64**3))
            data_i_label = dat_i.create_dataset("data_label", (length/2,),dtype='i8')
            data_i_set = X[s_data[i]]
            data_i_label = Y[s_data[i]]
            dat_i.close()
    
    dat.close()