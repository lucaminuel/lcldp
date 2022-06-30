# -*- coding: utf-8 -*-
import unittest
from machine_learning import basic_xgboost, hyper_xgboost
import pandas as pd

PATH = r'test_data.csv'

class TestXgbTool(unittest.TestCase):
    '''
    Unit test for Xgboost_tool
    '''
    def test_basic_xgboost(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, basic_xgboost, data = data, save = 5)


    def test_hyper_xgboost(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, hyper_xgboost, data = data, save = 5)
    
if __name__ == "__main__":
    unittest.main()
        


