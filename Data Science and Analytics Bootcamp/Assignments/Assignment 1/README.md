# Introduction to Python Assignment

## Overview

This repository contains my solutions for the “Introduction to Python” assignment. It demonstrates basic Python concepts including data types, control structures, functions, and classes.

## Objectives

* Practice working with Python’s built-in data types and collections  
* Implement simple functions (vowel counter, sum of evens)  
* Design and use a basic class (`BankAccount`)

## Prerequisites

* Python 3.6 or newer  
* (Optional) A code editor or IDE (e.g. VS Code, PyCharm)

## How to Run

1. Clone or download this repository.  
2. Open a terminal in the project folder.  
3. Run the assignment script:  
```

python assignment.py

```

## Task Descriptions

### Task 1: Data Types & Variables

* **What I did:** Created variables using int, float, string, bool, list, tuple, and dict types  
* **Why it matters:** Shows the ability to store and print various data types in Python

### Task 2: Functions & Classes

1. **`count_and_return_vowels(text)`**  
* Counts vowels (a, e, i, o, u) in a string (case-insensitive)  
* Returns a tuple: (count, list of vowels)

2. **`sum_of_even_numbers(limit)`**  
* Uses a `while` loop to sum even numbers up to the given limit  
* Returns the total sum

3. **`BankAccount` class**  
* Constructor to set initial balance  
* `deposit(amount)` to add funds  
* `withdraw(amount)` to remove funds (with insufficient funds check)  
* `get_balance()` to return the current balance

## Example Output

### Task 1
```

1
2.35
hello
True
\[1, 2, 3, 4, 5, 'abc']
(1, 2, 3, 4, 5, 'abc')
{'hameed': 'jasim', 'ali': 'husain', 'mohammed': 'sadiq'}

```

### Task 2
```

(3, \['e', 'o', 'o'])
(3, \['o', 'a', 'i'])
(4, \['O', 'e', 'A', 'I'])
30
6
0
100
150
120
Insufficient funds
120

```

## Notes & Improvements

* Added basic input validation in `BankAccount.withdraw()`  
* Could be extended with user inputs, error handling, or additional methods  
* Further ideas: add interest calculation or transaction logging features

---

**Author:** Jaffar Hasan  
**Date:** Feb 8, 2025
```
