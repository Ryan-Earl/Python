#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Ryan Earl
# Date: 10/20/2019
# E-mail: re022598@tamu.edu
# Description: This program creates a menu for you to add or edit fruit in a store

option = 0
inventory = {}

while option != 5:
    option = int(input("""******************* Main Menu *****************
1. Add fruits
2. Edit fruit
3. Delete fruit
4. Search fruits 
5. Quit 
***********************************************
Choose from menu: """))   

    if option == 1:
        new_fruit= input("Enter fruit followed by price: ").capitalize().split(',')
        inventory[new_fruit[0]] = [float(x) for x in new_fruit[1].split()]
        print(f"Current Stock: {inventory}")
        print("\n")
    elif option == 2:
        fruit_name = input("Enter the fruit name: ").capitalize()
        if fruit_name in inventory:
            fruit_prices = [float(x) for x in input("Enter the fruit prices: ").split()]
            inventory[fruit_name] = [fruit_prices]
            print(f"Current Stock: {inventory}")
            print("\n")
        else:
            print("This fruit is not in inventory")
            print("\n")
    elif option == 3:
        choice = input("If you want to remove all items enter all otherwise enter the fruit name: ").capitalize()
        if choice == 'All':
            inventory.clear()
            print(f"Current Stock: {inventory}")
            print("\n")
        else:
            if choice in inventory:
                del inventory[choice]
                print(f"Current Stock: {inventory}")
                print("\n")
            else:
                print("Fruit not in current stock, cannot delete")
                print("\n")
    elif option == 4:
        fruit_name = input("Enter the fruit name: ").capitalize()
        if fruit_name in inventory:
            formatted_values = str(inventory[fruit_name])[1:-1]
            print(f"{fruit_name} has {len(inventory[fruit_name])} prices: {formatted_values}")
            print("\n")
        else:
            print(f"{fruit_name} is not in the stock.")
            print(f"Current Stock: {inventory}")
            print("\n")
    elif option == 5:
        print(f"Current Stock: {inventory}")
    else:
        print("Invalid Chocie")

