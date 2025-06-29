# Introduction to Python Assignment

## Overview

A brief description of this repository’s purpose.
This repo contains my solutions for the “Introduction to Python” assignment, which demonstrates basic data types, control structures, functions, and classes in Python.

## Objectives

* Practice working with Python’s built-in data types and collections
* Implement simple functions (vowel counter, sum of evens)
* Design and use a basic class (`BankAccount`)

## Prerequisites

* Python 3.6 or newer
* (Optional) A code editor or IDE (e.g. VS Code, PyCharm)

## Repository Structure

.
├── assignment.py         # All tasks combined in one script
├── task1.py              # (Optional) Task 1: data types & variables
├── task2.py              # (Optional) Task 2: functions & classes
└── README.md             # This file

## How to Run

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. To run **all tasks** in one file:
   python assignment.py
4. Or run each task separately:
   python task1.py
   python task2.py

## Task Descriptions

### Task 1: Data Types & Variables

* **What you did:** Created variables of types int, float, string, bool, list, tuple, dict
* **Why it matters:** Demonstrates your ability to store and manipulate different kinds of data

### Task 2: Functions & Classes

1. **count\_and\_return\_vowels(text)**

   * Counts vowels (a, e, i, o, u) in the input string (case‑insensitive)
   * Returns a tuple (count, list\_of\_vowels)
2. **sum\_of\_even\_numbers(limit)**

   * Uses a while loop to sum even numbers up to limit
   * Returns the total sum
3. **BankAccount class**

   * Constructor to set initial balance
   * deposit(amount) to add funds
   * withdraw(amount) to remove funds (with insufficient‑funds check)
   * get\_balance() to retrieve current balance

## Example Output

# Task 1 prints:

1
2.35
hello
True
\[1, 2, 3, 4, 5, 'abc']
(1, 2, 3, 4, 5, 'abc')
{'hameed': 'jasim', 'ali': 'husain', 'mohammed': 'sadiq'}

# Task 2 prints:

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

## Notes & Improvements

* Added simple input validation in `BankAccount.withdraw()`
* Could extend with command‑line arguments or user input prompts
* Further practice: add interest calculation, transaction history, etc.

---

Author: Jaffar Hasan
Date: June 29, 2025
