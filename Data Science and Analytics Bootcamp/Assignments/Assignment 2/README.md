# Data Foundation with NumPy Assignment

## Overview

This assignment explores NumPy fundamentals through array operations, linear algebra, and advanced indexing. Tasks include working with student scores, reshaping and analyzing data, and applying matrix operations.

## Objectives

- Practice array creation, indexing, and boolean masking
- Use statistical and sorting functions with NumPy
- Perform matrix multiplication, inversion, and eigenvalue analysis
- Apply broadcasting and custom functions to real-world-like data

## Prerequisites

- Python 3.6 or higher
- NumPy installed (`pip install numpy`)

## How to Run

1. Make sure you have NumPy installed.
2. Save the script as `numpy_assignment.py`.
3. Run the script using:
   python numpy_assignment.py

## Tasks Covered

### === Array Operations and Indexing ===

**Task 1.1:** Calculate average score per student  
**Task 1.2:** Find highest score in each subject  
**Task 1.3:** Find students who scored above 90 in any subject  
**Task 1.4:** Boolean mask for students who passed all subjects (score >= 70)

### === Array Manipulation ===

**Task 2.1:** Reshape the scores array to 12x2  
**Task 2.2:** Create a standardized scores array (z-score normalization)  
**Task 2.3:** Sort students by average score in descending order  
**Task 2.4:** Use array methods to find min, max, and mean for each subject

### === Linear Algebra ===

**Task 3.1:** Multiply matrix_A and matrix_B using matrix multiplication  
**Task 3.2:** Calculate the determinant of matrix_A  
**Task 3.3:** Find the inverse of matrix_A (if it exists)  
**Task 3.4:** Calculate eigenvalues of matrix_A

### === Advanced Operations ===

**Task 4.1:** Use broadcasting to add 5 points to math scores  
**Task 4.2:** Find unique scores across all subjects  
**Task 4.3:** Boolean indexing to find students who scored above average in all subjects

### === Bonus Challenge ===

Created a function `student_report(student_name)` that returns:
- The student's individual scores
- Their rank in each subject
- A boolean indicating if they are in the top 3 overall

Example call:
student_report('Alice')

## Output Highlights

- Averages per student are printed
- Students are ranked and sorted
- Matrix operations show results (multiplication, inverse, eigenvalues)
- Function gives detailed report for a student

## Notes

- All code is commented and structured into logical blocks
- The script handles invertibility checks for matrix_A
- Uses NumPy functions such as mean(), std(), dot(), det(), inv(), eigvals(), unique(), and broadcasting

---

**Author:** Jaffar Hasan  
**Date:** Feb 26, 2025
