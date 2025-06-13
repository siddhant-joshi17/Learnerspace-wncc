mport numpy as np
import pandas as pd
import random
names = ["Student" + str(i) for i in range(10)]
subjects = random.choices(["Math", "Physics", "Chemistry"], k=10)
scores = np.random.randint(50, 101, size=10)
grades = []

def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

grades = [assign_grade(score) for score in scores]

students_df = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': grades
})

print("\nStudent DataFrame:")
print(students_df)

print("\nSorted by Score (descending):")
print(students_df.sort_values(by='Score', ascending=False))

print("\nAverage Score per Subject:")
print(students_df.groupby('Subject')['Score'].mean())

def pandas_filter_pass(df):
    return df[df['Grade'].isin(['A', 'B'])]

print("\nFiltered (Grades A & B):")
print(pandas_filter_pass(students_df))
