import os
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import h5py
from numba import jit, prange
from sklearn.model_selection import StratifiedKFold

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
    wrap = dsa.WrapDataObject(vtk_image)
    return adapter.PointData['ImageScalars'].reshape(dim,order='F')

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

def save_numpy(path_str,outpath):
    dat = h5py.File(path_str, 'r')
    X = np.array(dat.get('data'))
    Y = np.array(dat.get('data_label'))
    dat.close()

    np.save(os.path.join(outpath,'X.npy'),X)
    np.save(os.path.join(outpath,'Y.npy'),Y)


def split_data(output_path,name):

    X = np.load(os.path.join(output_path,'X.npy'),mmap_mode='r')
    Y = np.load(os.path.join(output_path,'Y.npy'),mmap_mode='r')

    skf = StratifiedKFold(n_splits=4, shuffle=True)
    i = 0
    for train_index, test_index in skf.split(X, Y):
        np.save(os.path.join(output_path,"x_{}_{}".format(name,i)), X[test_index])
        np.save(os.path.join(output_path,"y_{}_{}".format(name,i)), Y[test_index])
        i+=1
    