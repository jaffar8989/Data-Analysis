import numpy as np

# ---------------------------
# Setup: Creating arrays
# ---------------------------
# Student test scores for 3 subjects (math, science, english)
scores = np.array([
    [85, 92, 78],
    [90, 88, 95],
    [75, 70, 85],
    [88, 95, 92],
    [65, 72, 68],
    [95, 88, 85],
    [78, 85, 82],
    [92, 89, 90]
])

# Student names
names = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'])

# Random 4x4 matrices for linear algebra operations
matrix_A = np.random.randint(1, 10, size=(4, 4))
matrix_B = np.random.randint(1, 10, size=(4, 4))

print("=== Array Operations and Indexing ===\n")
# -------------------------------------------------
# Task 1.1: Calculate the average score for each student
# -------------------------------------------------
average_scores = scores.mean(axis=1)
print("Task 1.1: Average scores per student")
print("Average scores:", average_scores)
print("Students and their averages:")
for name, avg in zip(names, average_scores):
    print(f"{name}: {avg:.2f}")
print("\n")

# -------------------------------------------------
# Task 1.2: Find the highest score in each subject
# -------------------------------------------------
max_scores = scores.max(axis=0)
print("Task 1.2: Highest score in each subject (Math, Science, English):")
print(max_scores)
print("\n")

# -------------------------------------------------
# Task 1.3: Select all students who scored above 90 in any subject
# -------------------------------------------------
above_90_mask = scores > 90  # boolean mask for scores above 90
# Identify rows where at least one subject is above 90
students_above_90 = names[np.any(above_90_mask, axis=1)]
print("Task 1.3: Students who scored above 90 in any subject:")
print(students_above_90)
print("\n")

# -------------------------------------------------
# Task 1.4: Boolean mask for students who passed all subjects (passing score is 70)
# -------------------------------------------------
passing_mask = scores >= 70
# Select students where every subject meets the passing mark
students_passed_all = names[np.all(passing_mask, axis=1)]
print("Task 1.4: Students who passed all subjects (score >= 70):")
print(students_passed_all)
print("\n")

print("=== Array Manipulation ===\n")
# -------------------------------------------------
# Task 2.1: Reshape the scores array to be 12x2
# Note: The original scores array has 24 elements (8x3) so reshaping to 12x2 is valid.
# -------------------------------------------------
scores_12x2 = scores.reshape(12, 2)
print("Task 2.1: Reshaped scores array (12x2):")
print(scores_12x2)
print("\n")

# -------------------------------------------------
# Task 2.2: Create a new array with standardized scores
# Standardization is done by subtracting the overall mean and dividing by overall standard deviation.
# -------------------------------------------------
mean_all = scores.mean()
std_all = scores.std()
standardized_scores = (scores - mean_all) / std_all
print("Task 2.2: Standardized scores:")
print(standardized_scores)
print("\n")

# -------------------------------------------------
# Task 2.3: Sort the students by their average score in descending order
# -------------------------------------------------
sorted_indices = np.argsort(average_scores)[::-1]  # descending order indices
sorted_names = names[sorted_indices]
sorted_avg_scores = average_scores[sorted_indices]
print("Task 2.3: Students sorted by average score (descending):")
for name, avg in zip(sorted_names, sorted_avg_scores):
    print(f"{name}: {avg:.2f}")
print("\n")

# -------------------------------------------------
# Task 2.4: Use array methods to find min, max and mean for each subject
# -------------------------------------------------
min_scores = scores.min(axis=0)
max_scores_subject = scores.max(axis=0)
mean_scores_subject = scores.mean(axis=0)
print("Task 2.4: Per subject statistics:")
print("Minimum scores (Math, Science, English):", min_scores)
print("Maximum scores (Math, Science, English):", max_scores_subject)
print("Mean scores (Math, Science, English):", mean_scores_subject)
print("\n")

print("=== Linear Algebra ===\n")
# -------------------------------------------------
# Task 3.1: Multiply matrix_A and matrix_B using matrix multiplication
# -------------------------------------------------
product_matrix = np.dot(matrix_A, matrix_B)
print("Task 3.1: Matrix multiplication of matrix_A and matrix_B")
print("Matrix A:")
print(matrix_A)
print("Matrix B:")
print(matrix_B)
print("Product (A * B):")
print(product_matrix)
print("\n")

# -------------------------------------------------
# Task 3.2: Calculate the determinant of matrix_A
# -------------------------------------------------
det_A = np.linalg.det(matrix_A)
print("Task 3.2: Determinant of matrix_A:")
print(det_A)
print("\n")

# -------------------------------------------------
# Task 3.3: Find the inverse of matrix_A (if it exists)
# We check the determinant to ensure matrix_A is invertible.
# -------------------------------------------------
print("Task 3.3: Inverse of matrix_A (if invertible):")
if np.abs(det_A) > 1e-6:
    inv_A = np.linalg.inv(matrix_A)
    print(inv_A)
else:
    print("matrix_A is not invertible.")
print("\n")

# -------------------------------------------------
# Task 3.4: Calculate the eigenvalues of matrix_A
# -------------------------------------------------
eigenvalues = np.linalg.eigvals(matrix_A)
print("Task 3.4: Eigenvalues of matrix_A:")
print(eigenvalues)
print("\n")

print("=== Advanced Operations ===\n")
# -------------------------------------------------
# Task 4.1: Use broadcasting to add 5 points to all math scores (first column)
# -------------------------------------------------
scores_with_bonus = scores.copy()
scores_with_bonus[:, 0] += 5
print("Task 4.1: Scores after adding 5 bonus points to math scores:")
print(scores_with_bonus)
print("\n")

# -------------------------------------------------
# Task 4.2: Find unique scores across all subjects
# -------------------------------------------------
unique_scores = np.unique(scores)
print("Task 4.2: Unique scores across all subjects:")
print(unique_scores)
print("\n")

# -------------------------------------------------
# Task 4.3: Use boolean indexing to find students who scored above average in all subjects
# We calculate the mean for each subject, then check which students have scores greater than these means.
# -------------------------------------------------
subject_means = scores.mean(axis=0)
above_avg_mask = scores > subject_means  # boolean mask per element
students_above_avg_all = names[np.all(above_avg_mask, axis=1)]
print("Task 4.3: Students who scored above average in all subjects:")
print(students_above_avg_all)
print("\n")

print("=== Bonus Challenge ===\n")
# -------------------------------------------------
# Bonus Challenge: Function to report individual student performance details
# -------------------------------------------------
def student_report(student_name):
    """
    Given a student's name, print:
      - Their individual scores for Math, Science, and English.
      - Their ranking in each subject (1 = highest).
      - A boolean indicating if they are in the top 3 performers overall (by average score).
    """
    if student_name not in names:
        print("Student not found.")
        return None
    
    # Get the index for the student in the arrays.
    idx = np.where(names == student_name)[0][0]
    individual_scores = scores[idx]
    
    # Calculate ranking in each subject
    subjects = ['Math', 'Science', 'English']
    rankings = {}
    for i, subject in enumerate(subjects):
        # Get scores for this subject
        subject_scores = scores[:, i]
        # Argsort in descending order gives ranking (position + 1)
        sorted_indices_subject = np.argsort(-subject_scores)
        rank = np.where(sorted_indices_subject == idx)[0][0] + 1
        rankings[subject] = rank

    # Overall ranking based on average score
    overall_sorted_indices = np.argsort(-average_scores)
    overall_rank = np.where(overall_sorted_indices == idx)[0][0] + 1
    top3_overall = overall_rank <= 3

    # Display the report
    print(f"Report for {student_name}:")
    print("Individual Scores (Math, Science, English):", individual_scores)
    print("Rankings in each subject:", rankings)
    print("Overall ranking (by average score):", overall_rank)
    print("Is top 3 performer overall?", top3_overall)
    
    return individual_scores, rankings, top3_overall

# Example call to the function for demonstration
student_report('Alice')


