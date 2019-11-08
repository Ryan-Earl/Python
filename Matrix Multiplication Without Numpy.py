#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: Ryan Earl
# Date: 10/13/2019
# E-mail: re022598@tamu.edu
# Description: This program creates two matrices given the dimensions, multiplies them and creates the transpose matrix

row_a, col_a = map(int, input("Enter matrix A rows and columns: ").split())
row_b, col_b = map(int, input("Enter matrix B rows and columns: ").split())
row_c, col_c = row_a, col_b

if(row_a >10 or row_b > 10 or col_a > 10 or col_b > 10):
    print("ERROR: the number of rows or columns cannot exceed 10")
    
elif(col_a != row_b):
    print("ERROR: the number of columns in matrix A must equal the number of rows in matrix B")
    
else:
    # Convert input values into a list of float values, then split into rows and columns
    a_values = [float(x) for x in (input("Enter matrix A: ").split())]
    matrix_a = [a_values[x:x+col_a] for x in range(0, len(a_values), col_a)]
    b_values = [float(x) for x in (input("Enter matrix B: ").split())]
    matrix_b = [b_values[x:x+col_b] for x in range(0, len(b_values), col_b)]
    
    #create an empty matrix for c
    matrix_c = [[0*x for x in range(col_c)] for y in range(row_c)]
    
    # populate matrix c
    for x in range(row_a):
        for y in range(col_b):
            dot_product = 0
            for z in range(col_a):
                dot_product = dot_product + (matrix_a[x][z] * matrix_b[z][y])
            matrix_c[x][y] = dot_product
            
    # create and populate the transpose matrix T
    matrix_t = [[0*x for x in range(row_c)] for y in range(col_c)]
    index = 0
    for x in range(len(matrix_c[0])):
        for x in range(len(matrix_c)):
            matrix_t[index][x] = matrix_c[x][index]
        index += 1
    
    #create a function to print a matrix
    def PrintMatrix(matrix):
        for row in matrix:
            correct_type = [int(x) for x in row]
            for element in correct_type:
                if element >= 10:
                    print(element, end='  ')
                else:
                    print(element, end='   ')
            print()
    
    print()
    print("Matrix A: ")
    PrintMatrix(matrix_a)
    print()
    print("Matrix B: ")
    PrintMatrix(matrix_b)
    print()
    print("Matrix C: ")
    PrintMatrix(matrix_c)
    print()
    print("Transpose matrix T: ")
    PrintMatrix(matrix_t)

