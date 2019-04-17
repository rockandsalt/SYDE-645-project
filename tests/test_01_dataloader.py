import unittest
import utils.dataloader
from vtk.numpy_interface import dataset_adapter as dsa
import os
import vtk
import numpy as np
import h5py

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.path_file = os.path.join('.','dataset','stl','23_6sides_pocket','23_30.STL')
        self.output_path = os.path.join('.','tests','test_output')

    def test_path(self):
        path = utils.dataloader.sampler(3)
        self.assertTrue(os.path.isfile(path[0][0]))
    
    def test_load_data(self):
        vol = utils.dataloader.load_file(self.path_file)

        output_path = os.path.join(self.output_path,'23_30.vti')
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(vol)

        writer.Write()
    
    def test_convert_numpy(self):
        vol = utils.dataloader.load_file(self.path_file)

        dim = vol.GetDimensions()
        
        arr = utils.dataloader.convert_image_to_numpy(vol)
        
        self.assertEqual(arr.shape[0],dim[0])
        self.assertEqual(arr.shape[1],dim[1])
        self.assertEqual(arr.shape[2],dim[2])
    
    def test_convert_data(self):
        utils.dataloader.convert_all_data(self.output_path)
    
    def test_convert_to_np(self):
        utils.dataloader.save_numpy(os.path.join(self.output_path,'data.hdf5'),self.output_path)
    
    def test_read_hdf5(self):
        path = os.path.join(self.output_path,'data.hdf5')
        dat = h5py.File(path, 'r')

        x = dat.get('data')
        y = dat.get('data_label')

        x_0 = np.array(x[0])
        x_20 = np.array(x[2000])
        print(np.max(x_20))
        
        self.assertTrue(not np.array_equal(x_0,x_20))
        #self.assertEqual(y_0,y_20)

        dat.close()
    
    def test_split_data(self):
        utils.dataloader.split_data(self.output_path,'split8')