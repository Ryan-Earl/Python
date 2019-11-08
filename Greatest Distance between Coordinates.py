#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Author: Ryan Earl
# Date: 7/7/2019
# E-mail: re022598@tamu.edu
# Description: this script tells you the distance between the closest and furthest points

source = [float (x) for x in input("Enter source coordinates: ").split()]
a = [float (x) for x in input("Enter point A coordinates: ").split()]
b = [float (x) for x in input("Enter point B coordinates: ").split()]
c = [float (x) for x in input("Enter point C coordinates: ").split()]
points = [a, b, c]
locations = ("A", "B", "C")

d = []
for x in points:
    d.append(((source[0] - x[0])**2 + (source[1] - x[1])**2)**.5) 

max_dis = 0
for x in d:
    if x > max_dis:
        max_dis = x
max_ind = d.index(max_dis)
    
min_dis = max_dis
for x in d:
    if x < min_dis:
        min_dis = x
min_ind = d.index(min_dis)

print()
print(f"From source point {source[0], source[1]},")
print(f"Point {locations[min_ind]} {points[min_ind]} is closest, with distance of {round(min_dis, 2)} units.")
print(f"Point {locations[max_ind]} {points[max_ind]} is closest, with distance of {round(max_dis, 2)} units.")

