#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: Ryan Earl
# Date: 11/05/2019
# E-mail: re022598@tamu.edu
# Description: This script creates team rosters in order seperate a class into teams. Students follow Student, UIN format

import random
import math
option = 110
new_team = True

# define functions

# print completed team rosters
def print_teams(dictionary):
    count = 1
    print("Current Teams: ")
    
    if len(dictionary) <= 0:
        print("There are no remaining teams. ")
    
    else:
        #sort through nested loops and print values
        for team in dictionary.values():
            print(f"Team {count}: ")
            for kid, id_num in team.items():
                if len(kid) >= 7:
                    print(f"{kid}"+ '  ' +f"{id_num}")
                else:
                    print(f"{kid} \t {id_num}")
            print()
            count += 1

# validate and create team dictionary
def validate_teams(roster):
    global team_size, team_num
    
    # get team size and number of students per team
    team_size = int(input("What is the team size: "))
    team_num = math.ceil(len(roster)/team_size)
        
    # check to ensure team size does not exceed student count
    while team_size > len(roster):
        print("Invalid team size, team size cannot exceed number of students. Try again")
        team_size = int(input("What is the team size: "))
    print()
    
    # create dictionary of teams by team number
    teams = {}
    for x in range(1,team_num+1):
        teams["team_{0}".format(x)]= {}
    
    return teams

while option != 6:
    option = int(input("""******************* Main Menu *****************
1. Create roster
2. Create teams manually
3. Create teams randomly
4. Sort team members 
5. Delete a team
6. Exit
***********************************************

Choose a menu option: """))
    
    if option == 1:
        students = int(input("how many students would you like to enter: "))
        roster = {}
        
        while len(roster) < students:
            
            # get student name and UIN
            try:
                student, uin = input(f"Enter student {len(roster)+1}: ").split(" ")
                student = student.capitalize()
            except ValueError:
                print("Invalid input, make sure both UIN and student name are included")
                continue
                
            # check 1: student is not repeated in roster, if so ask for last initial
            if student in roster:
                student += input(f"Enter the last initial for {student}").capitalize()
                
            # check 2: is the student's name alphabetic
            if student.isalpha() == False:
                print("Name contains invalid characters, try again.")
                continue
                
            # check 3: is the UIN numeric, 9 digits, and unique
            elif (uin.isdigit() == False or len(uin) != 9 or uin in roster.values()):
                print("UIN is invalid, try again.")
                continue
                
            roster[student] = uin
            
        print()
        print("Current roster: ")  
        print("Student Name \t UIN")
        roster_copy = roster
        
        for key, value in roster.items():
            if len(key) >= 7:
                print(f"{key} \t {value}")
            else:
                print(f"{key} \t\t {value}")
        print()
                       
    elif option == 2:
        
        if new_team == False:
            print("Invalid choice, teams have already been created")
            print()
            continue
        
        else:
            # create nested dictionaries and populate dictionary of teams
            teams = validate_teams(roster)
            team_count = 1
        
            for key in teams.keys():
                member = 1
                print(f"Enter students for team {team_count}: ")
                while member <= team_size:
                
                    # if there are no students left to choose, exit
                    if bool(roster_copy) == False:
                        break
                    try:
                        name, id_num = input().split(" ")
                        name = name.capitalize()
                    except ValueError:
                        print("Invalid input, make sure both UIN and student name are included")
                        continue
                    
                    #create multiline input, check to make sure students exist in roster
                    if name:
                        if name in roster_copy:
                            if roster[name] == id_num:
                                teams[key][name] = id_num
                                # remove student once they have been selected for a team
                                del roster_copy[name]
                            else:
                                print("incorrect ID number, try again.")
                                continue
                        else: 
                            print("Student not in roster, try again.")
                            continue
                        member += 1
                    else:
                        print("You must enter a student, try again. ")
                team_count += 1
                print()
            
            print_teams(teams)
            new_team = False
        
    elif option == 3:
        
        if new_team == False:
            print("Invalid choice, teams have already been created")
            print()
            continue
            
        else:
            #create nested dictionary of teams
            teams = validate_teams(roster)
            
            #randomly select individuals in roster to add to teams
            for key in teams.keys():
                member = 1
                while member <= team_size:
                    
                    # if there are no students left to choose, exit
                    if bool(roster_copy) == False:
                        break
                    #populate created team dictionary
                    name, id_num = random.choice(list(roster.items()))
                    teams[key][name] = id_num
                    # remove student once they have been selected for a team
                    del roster_copy[name]
                    member += 1
                    
            print()
            print_teams(teams)
            new_team = False
    
    elif option == 4:
    
        order = 'none'
        acceptable_responses = {'ascending','descending'}
        while order not in acceptable_responses:
        
            order = input("Sort team members in ascending or descending order of UIN? ").lower()
            if order not in acceptable_responses:
                print("Input Invalid, must be either 'ascending' or 'descending'")
    
        print()
            
        if order == 'ascending':
            #sort and print teams in ascending order
            team_count = 1
            print("Current Teams")
            for key in teams.keys():
                print(f"Team {team_count}: ")
                for student, uin in sorted(teams[key].items(), reverse=True):
                    if len(student) >= 7:
                        print(f"{student}"+ '  ' +f"{teams[key][student]}")
                    else:
                        print(f"{student} \t {teams[key][student]}")
                team_count += 1
                print()
        else:
            #sort and print teams in descending order
            team_count = 1
            print("Current Teams")
            for key in teams.keys():
                print(f"Team {team_count}: ")
                for student, uin in sorted(teams[key].items()):
                    if len(student) >= 7:
                        print(f"{student}"+ '  ' +f"{teams[key][student]}")
                    else:
                        print(f"{student} \t {teams[key][student]}")
                team_count += 1
                print()
                
    elif option == 5:
        
        # get input team and delete them from the dictionary
        to_delete = input("Which team would you like to delete? ")
        print()
        if len(teams) > 1:
            match = [x for x in teams.keys() if to_delete in x]
            del teams[match[0]]
        elif len(teams) == 1:
            teams.clear()
        print_teams(teams)

