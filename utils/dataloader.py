import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

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
    