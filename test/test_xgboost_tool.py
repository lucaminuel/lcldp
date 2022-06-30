# -*- coding: utf-8 -*-
import unittest
from machine_learning.xgboost_tool import basic_xgboost as bxgb
from machine_learning.xgboost_tool import hyper_xgboost as hxgb
import pandas as pd

PATH = r'test_data.csv'

class TestXgbTool(unittest.TestCase):
    '''
    Unit test for Xgboost_tool
    '''
    def test_basic_xgboost(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, bxgb, data = data, save = 5)


    def test_hyper_xgboost(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, hxgb, data = data, save = 5)
    
if __name__ == "__main__":
    unittest.main()
        


