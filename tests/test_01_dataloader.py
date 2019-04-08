import unittest
import utils.dataloader
import os
import vtk
import numpy as np
import h5py

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.path_file = os.path.join('.','dataset','stl','0_Oring','0_1.STL')
        self.output_path = os.path.join('.','tests','test_output')

    def test_path(self):
        path = utils.dataloader.sampler(3)
        self.assertTrue(os.path.isfile(path[0][0]))
    
    def test_load_data(self):
        vol = utils.dataloader.load_file(self.path_file)

        output_path = os.path.join(self.output_path,'0_1.vti')
        
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
    
    def test_read_hdf5(self):
        path = os.path.join(self.output_path,'data.hdf5')
        dat = h5py.File(path, 'r')

        x = dat.get('data')

        x_0 = np.array(x[0])
        x_20 = np.array(x[20])
        
        self.assertTrue(not np.array_equal(x_0,x_20))

        dat.close()