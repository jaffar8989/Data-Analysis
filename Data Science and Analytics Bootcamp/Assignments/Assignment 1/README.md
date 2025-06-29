```
# Python Fundamentals Assignment

## Overview
This repository contains solutions for the “Introduction to Python” assignment. It demonstrates basic Python concepts including variables, data types, control flow, functions, and object‑oriented programming.

## Prerequisites
- Python 3.6 or higher installed on your system

## Files
- `main.py`  
  Contains all Task1–Task3 solutions in a single script.

## How to Run
1. Open a terminal (or command prompt).
2. Navigate to the folder containing `main.py`.
3. Run:
```

python main.py

````
4. Observe the printed outputs for each task in your console.

---

## Task 1: Basic Data Types and Printing

In `main.py`, Task 1 shows how to create and print variables of different data types:

```python
x = 1                   # integer
y = 2.35                # float
z = "hello"             # string
w = True                # boolean

list1 = [1, 2, 3, 4, 5, 'abc']
tuple1 = (1, 2, 3, 4, 5, 'abc')
dict1 = {'hameed': 'jasim', 'ali': 'husain', 'mohammed': 'sadiq'}

print(x)
print(y)
print(z)
print(w)
print(list1)
print(tuple1)
print(dict1)
````

---

## Task 2: Functions

### 1. `count_and_return_vowels(text)`

* **Input:** a string `text`
* **Output:** a tuple `(count, vowels_list)`
* **Description:** Counts how many vowels are in the string and returns both the count and a list of found vowels.

Example usage:

```python
print(count_and_return_vowels("Hello World"))   # (3, ['e', 'o', 'o'])
print(count_and_return_vowels("Programming"))   # (3, ['o', 'a', 'i'])
print(count_and_return_vowels("OpenAI"))        # (4, ['O', 'e', 'A', 'I'])
```

### 2. `sum_of_even_numbers(limit)`

* **Input:** an integer `limit`
* **Output:** the sum of all even numbers from 2 up to `limit`.
* **Description:** Uses a `while` loop to accumulate even numbers.

Example usage:

```python
print(sum_of_even_numbers(10))  # 30
print(sum_of_even_numbers(5))   # 6
print(sum_of_even_numbers(1))   # 0
```

---

## Task 3: Object‑Oriented Programming

Defines a `BankAccount` class with:

* **Constructor:** `__init__(self, initial_balance=0)`
* **Methods:**

  * `deposit(amount)`: Add funds if `amount > 0`.
  * `withdraw(amount)`: Subtract funds if sufficient balance; otherwise prints an error.
  * `get_balance()`: Returns the current balance.

Example usage:

```python
account = BankAccount(100)
print(account.get_balance())   # 100

account.deposit(50)
print(account.get_balance())   # 150

account.withdraw(30)
print(account.get_balance())   # 120

account.withdraw(200)          # Prints "Insufficient funds"
print(account.get_balance())   # 120
```

---

## Notes

* Make sure your Python path is correctly set.
* Feel free to split each task into separate modules or files for better organization.
* This script is meant for learning and demonstration purposes only.

```
```
