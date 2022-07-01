# -*- coding: utf-8 -*-
import unittest
from data_analysis.plot_image import barpl, histpl,countpl,corr_matrix
import pandas as pd

PATH = r'./test_data.csv'

class TestLoadInfoData(unittest.TestCase):
    '''
    Unit test for plot_image
    '''
    def test_countpl(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, countpl,column = 'loan_status' , data = data, save = 5)
    
    def test_barpl(self):
        data = pd.read_csv(PATH)
        grade_co = data[data['loan_status'] == 'Charged Off'].groupby('grade').count()['loan_status']
        grade_fp = data[data['loan_status'] == 'Fully Paid'].groupby('grade').count()['loan_status']
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, barpl, column_co= grade_co, column_fp = grade_fp , save = 5)
      
    def test_histpl(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, histpl,column = 'int_rate' , data = data, save = 5)

    def test_corr_matrix(self):
        data = pd.read_csv(PATH)
        #So that save is not in (0, 1), there is error
        self.assertRaises(ValueError, corr_matrix, data = data, save = 5)
    
if __name__ == "__main__":
    unittest.main()
        


