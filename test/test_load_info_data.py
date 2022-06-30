# -*- coding: utf-8 -*-
import unittest
import numpy as np
import data_analysis
import pandas as pd

PATH = r'test_data.csv'


class TestLoadInfoData(unittest.TestCase):
    '''
    Unit test for load_info_data module
    '''
    def test_load_info_data(self):
        data = data_analysis.load_info_data(PATH)
        #I evaluate the sum of some columns
        sum_loan = data['loan_amnt'].sum(axis=0)
        sum_mort_atc = data['mort_acc'].sum(axis=0)
        sum_total_acc = data['total_acc'].sum(axis=0)
        true_sum = np.array([124175.0, 10, 227])
        self.assertEqual(sum_loan,true_sum[0])
        self.assertEqual(sum_mort_atc,true_sum[1])
        self.assertEqual(sum_total_acc,true_sum[2])



if __name__ == "__main__":
    unittest.main()
        
    

