#Task1
x=1
print(x)
y=2.35
print(y)
z="hello"
print(z)
w=True
print(w)
list1=[1,2,3,4,5,'abc']
print(list1)
tuple1=(1,2,3,4,5,'abc')
print(tuple1)
dict1={'hameed':'jasim','ali':"husain",'''mohammed''':"sadiq"}
print(dict1)
#Task2
def count_and_return_vowels(text):
    vowels = [char for char in text if char.lower() in 'aeiou']
    return len(vowels), vowels


print(count_and_return_vowels("Hello World")) # output: (3, ['e', 'o', 'o'])
print(count_and_return_vowels("Programming")) # output: (3, ['o', 'a', 'i'])
print(count_and_return_vowels("OpenAI")) # output: (4, ['O', 'e', 'A', 'I'])


def sum_of_even_numbers(limit):
    total = 0
    i = 2
    while i <= limit:
        total += i
        i += 2
    return total


print(sum_of_even_numbers(10)) # output: 30
print(sum_of_even_numbers(5)) # output: 6
print(sum_of_even_numbers(1)) # output: 0


class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount

    def withdraw(self, amount):
        if amount < 0:
            print("Invalid withdrawal amount")
            return False
        elif amount <= self.balance:
            self.balance -= amount
            return True
        else:
            print("Insufficient funds")
            return False

    def get_balance(self):
        return self.balance


account = BankAccount(100)
print(account.get_balance())  # output: 100
account.deposit(50)
print(account.get_balance())  # output: 150
account.withdraw(30)
print(account.get_balance())  # output: 120
account.withdraw(200)  # Should print: "Insufficient funds"
print(account.get_balance())  # output: 120


