import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import h5py

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
    vox_modeller.SetScalarTypeToFloat()
    vox_modeller.SetMaximumDistance(.1)

    vox_modeller.SetInputConnection(stl_reader.GetOutputPort())
    vox_modeller.Update()
    vox_modeller.GetOutput()

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
            data_set[path_id+label_id] = arr.reshape(64**3)
            data_label[path_id+label_id] = label_id
            hf.flush()

    hf.close()
    
    