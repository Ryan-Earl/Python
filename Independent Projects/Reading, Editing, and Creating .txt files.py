#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def num_grade(scores):
    return round((scores[0]*.25)+(scores[1]*.35)+(scores[2]*.40), 1)

def letter_grade(scores):
    # converts numerical grade into letter grade
    if type(scores) not in (int, float):
        grade = num_grade(scores)
    else:
        grade = scores
    if grade >= 90: return "A"
    elif 90 > grade >= 80: return "B"
    elif 80 > grade >= 70: return "C"
    elif 70 > grade >= 60: return "D"
    else: return "F"

def main():
    
    # open exam_scores.txt, read file, strip rows and convert each to a list
    exam_scores = []
    in_file = open(r"C:\Users\Eileen\Documents\exams\exam_scores.txt", "r")
    scores = in_file.readline().strip()
    while scores != '':
        exam_scores.append(scores)
        scores = in_file.readline().strip()
    in_file.close()
    
    # convert exam_scores to nested list of floats, 1 for each student
    exam_scores = [[float(i) for i in exam_scores[student].split()] for student in range(len(exam_scores))]
    
    # create new file, write to it and close
    grades = open(r"C:\Users\Eileen\Documents\exams\grades.txt", "w")
    for i in range(len(exam_scores)):
        grades.write(f"ID:{int(exam_scores[i][0])} has a score of {num_grade(exam_scores[i][1:])}, Letter grade : {letter_grade(exam_scores[i][1:])}\n")
    grades.close()
    
    # create dictionary of student ids to grades
    ids = {}
    for i in range(len(exam_scores)):
        ids[exam_scores[i][0]] = exam_scores[i][1:]
     
    # output sorted ids to .txt file
    sorted_ids = open(r"C:\Users\Eileen\Documents\exams\sorted_ids.txt", "w")
    for key, value in sorted(ids.items()):
        sorted_ids.write(f"ID:{int(key)} has a score of {num_grade(value)}, Letter grade : {letter_grade(value)}\n")
    sorted_ids.close()
    
    # output sorted graes to .txt file
    sorted_grades = open(r"C:\Users\Eileen\Documents\exams\sorted_grades.txt", "w")
    grades = {}
    for key, value in ids.items():
        grades[num_grade(value)] = key
    for key, value in sorted(grades.items()):
        sorted_grades.write(f"ID:{int(value)} has a score of {key}, Letter grade : {letter_grade(key)}\n")
    sorted_grades.close
main()

