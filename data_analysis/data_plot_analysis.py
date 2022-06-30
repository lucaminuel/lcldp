# -*- coding: utf-8 -*-
#pylint: disable=line-too-long
#pylint: disable=wildcard-import
#pylint: disable=unused-import
#pylint: disable=unused-wildcard-import
# Copyright 2022 Manuel Luci-Andrea Dell'Abate
#
# This file is part of CMEPDA Project
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
In this first module  of data analysis we analyze our dataset and make useful plot
"""
import time
import seaborn as sns
import data_analysis
from plot_image import *
PATH =  r'..\lending_club_loan_two.csv'

if __name__ == '__main__':
    #Import dataset and print it's info
    start = time.time()
    data = data_analysis.load_info_data(PATH)
    #Visualizing loan payoff and chargeoff
    countpl(column = 'loan_status' , data = data,  save = 0, name = '(1)pf_vs_co.pdf')
    #Compute Matrix Correlation and plot it
    corr_matrix(data,  save = 0, name ='(2)corr_matrix.pdf')
    #Visualize relationship between grade and loan status
    sorted_grade = sorted(data['grade'].unique())
    countpl(column = 'grade', data = data, hue= 'loan_status',  save = 0, \
            name = '(3)grade.pdf', order = sorted_grade)
    #Grade is a good indicator if the borrower has the ability to payoff or not
    grade_co = data[data['loan_status'] == 'Charged Off'].groupby('grade').count()['loan_status']
    grade_fp = data[data['loan_status'] == 'Fully Paid'].groupby('grade').count()['loan_status']
    barpl(grade_co, grade_fp,  save = 0, name = '(4)grade_bar.pdf')
    #Visualize relationship between subgrade and loan status
    sorted_sub_grade = sorted(data['sub_grade'].unique())
    countpl(column = 'sub_grade', data = data, hue= 'loan_status',  save = 0, \
            name = '(5)sub_grade.pdf', order = sorted_sub_grade)
    sub_grade_co = data[data['loan_status'] ==
                        "Charged Off"].groupby("sub_grade").count()['loan_status']
    sub_grade_fp = data[data['loan_status'] ==
                        'Fully Paid'].groupby('sub_grade').count()['loan_status']
    #They give more informations than grade
    barpl(sub_grade_co, sub_grade_fp,  save = 0, name = '(6)sub_grade_bar.pdf')
    #Visualize relationship between  loan_status and verification_status
    sorted_verification_status = sorted(data["verification_status"].unique())
    countpl(column = 'verification_status' , data = data, hue='loan_status',  save = 0, \
            name = '(7)verification_status.pdf')
    #Source Verified and Verified have almost same information
    verification_status_co = data[data["loan_status"] == "Charged Off"].groupby("verification_status").count()["loan_status"]
    verification_status_fp = data[data["loan_status"] == "Fully Paid"].groupby("verification_status").count()["loan_status"]
    barpl(verification_status_co, verification_status_fp,  save = 0,\
          name = '(8)verification_status_bar.pdf')
    #Visualize relationship between int_rate and loan_status
    sorted_int_rate = sorted(data['int_rate'].unique())
    histpl(column = 'int_rate', data = data,  save = 0, \
           name = '(9)int_rate_hist.pdf', hue = 'loan_status', binwidth= 2)
    #Visualizie relationship betweeen emp_length and loan_status
    sorted(data["emp_length"].dropna().unique())
    sorted_emp_length = ['< 1 year', '1 year', '2 years', '3 years', '4 years',\
                         '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
    countpl(column = 'emp_length', data = data, hue= 'loan_status',  save = 0, \
            order=sorted_emp_length, name = '(10)emp_lenght.pdf')
    #emp_length doesn't help in ditinguish borrowers who payoff or not
    emp_co = data[data["loan_status"] == "Charged Off"].groupby("emp_length").count()["loan_status"]
    emp_fp = data[data["loan_status"] == "Fully Paid"].groupby("emp_length").count()["loan_status"]
    barpl(emp_co, emp_fp,  save = 0,\
          name = '(11)emp_lenght_bar.pdf')
    #Visualize relationship between  purprose and loan_status
    countpl(column = 'purpose', data = data, hue= 'loan_status',  save = 0, \
            name = '(12)purpose.pdf')
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    data.info()
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
    