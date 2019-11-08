#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib
from urllib.request import urlopen
import math

data = urllib.request.urlopen('https://github.tamu.edu/raw/TAMU-GEOG-676-GIS-Programming/Code/master/code/03/shapes.txt?token=AAAkKfeXd0OUXL71imhPk572v8DXNwzVks5djirgwA%3D%3D').read()
data = list(item.split(',') for item in data.decode('utf-8').split('\n'))[:-1]

class Rectangle():
    
    def setLength(self, length):
        self.l = float(length)
    
    def setWidth(self, width):
        self.w = float(width)
    
    def getArea(self, l, w):
        self.area = l*w
    
class Circle():
    
    def setRadius(self, radius):
        self.r = float(radius)
    
    def c_getArea(self, r):
        self.area = math.pi * self.r * self.r

class Triangle():
    
    def setHeight(self, height):
        self.h = float(height) 
    
    def setBase(self, base):
        self.b = float(base)
    
    def t_getArea(self, b, h):
        self.area = (b*h)/2
        shape_count = 1

shape_count = 1
shape_areas = {}

for shape in data:
    if shape[0] == 'Rectangle':
        shape_x = Rectangle()
        shape_x.setLength(shape[1])
        shape_x.setWidth(shape[2])
        shape_x.getArea(shape_x.l,shape_x.w)
        shape_areas.update({shape_count : shape_x.area})
    elif shape[0] == 'Circle':
        shape_x = Circle()
        shape_x.setRadius(shape[1])
        shape_x.c_getArea(shape_x.r)
        shape_areas.update({shape_count : shape_x.area})
    else:
        shape_x = Triangle()
        shape_x.setHeight(shape[1])
        shape_x.setBase(shape[2])
        shape_x.t_getArea(shape_x.b, shape_x.h)
        shape_areas.update({shape_count : shape_x.area})
    shape_count += 1
    
for x in range(0,len(data)):
    print(f"Shape Type: {data[x][0]}, Shape number {x+1}, Shape area: {shape_areas.get(x+1)}")

