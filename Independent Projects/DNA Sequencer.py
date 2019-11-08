#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Author: Ryan Earl
# Date: 10/20/2019
# E-mail: re022598@tamu.edu
# Description: This program finds all the sequences of dna in a set and returns the most frequent pattern

dna = input("Enter DNA sequence: ")
pattern_length = int(input("Enter pattern length: "))
sequenced_dna = [dna[x:x+pattern_length] for x in range(0, len(dna)) if len(dna[x:x+pattern_length]) == pattern_length]
print()

sequence_counts = {}
for seq in set(sequenced_dna):
    sequence_counts[seq] = sequenced_dna.count(seq)

print(f"Most frequent pattern of length {pattern_length}:")
for key, value in sequence_counts.items():
    if value == max(sequence_counts.values()):
        print(key)

