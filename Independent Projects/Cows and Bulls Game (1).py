#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Author: Ryan Earl
# Date: 7/21/2019
# E-mail: re022598@tamu.edu
# Description: this script plays the game cows and bulls based on user input

import random

def welcome():
    print('-----------------------------')
    print('| Welcome to Cows and Bulls |')
    print('-----------------------------')
    print()
    
def cowsandbulls():
    print()
    secret_num = str(random.randint(100, 999))
    secret_list = [int(x) for x in secret_num]
    cows = 0
    bulls = 0
    ind = 0
    guess_num = 1
    
    while cows < 3:
        guess = str(input(f"Guess #{guess_num}: "))
        guess_list = [int(x) for x in guess]
        
        cows = 0
        bulls = 0
                  
        for x in guess_list:
            if x == secret_list[ind]:
                cows += 1
            else:
                bulls += 1
            ind +=1
        
        ind = 0    
        guess_num += 1
        print(f"{cows} cows, {bulls} bulls")
    
    print()
    print("You got it!")
    print(f"it took you {guess_num-1} guesses to guess the secret number {secret_num}")
        

def main():
    
    welcome()
    cowsandbulls()

main()

