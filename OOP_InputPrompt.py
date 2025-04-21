from abc import ABC, abstractmethod

# C. Abstract Base Class
class Calculation(ABC):
    @abstractmethod
    def calculate(self, a, b):
        pass

# B. Operation Classes to inherit from calculation class
class Addition(Calculation):
    def calculate(self, a, b):
        return a + b

class Subtraction(Calculation):
    def calculate(self, a, b):
        return a - b

class Multiplication(Calculation):
    def calculate(self, a, b):
        return a * b

class Division(Calculation):
    def calculate(self, a, b):
        return a / b if b != 0 else "Cannot divide by zero"

class Modulus(Calculation):
    def calculate(self, a, b):
        return a % b

class DotProduct(Calculation):
    def calculate(self, a, b):
        dot_sum = 0
        if len(a) == len(b):
            # for i in range(len(a)):
            #     a[i] = int(a[i]) 
            a = [int(i) for i in a]
            b = [int(i) for i in b]
            for i in range(len(a)):
                dot_sum += (a[i]*b[i])
            return dot_sum
        else:
            print("Length of list most be equal")
# A. Top-level object of a class Calculator Class
class Calculator:
    def __init__(self):
        self.operations = {'+': Addition(),'-': Subtraction(),'*': Multiplication(),'/': Division(),'%': Modulus(),'dp': DotProduct() }

    def do_calculation(self, operation, a, b):
        if operation in self.operations:
            return self.operations[operation].calculate(a, b)
        else:
            return "Operation not supported"

if __name__ == "__main__":
    calc = Calculator()
    input_opr = int(input("Which operation do you want to perform? 0 = basic operations 1 = dot produc: "))
    if input_opr == 0:
        input_value1 = int(input("Please input your first degit: "))
        input_value2 = int(input("Please input your second degit: "))
        input_value = input("Select one of the following Operations you want to perform (+,-,*,/,%): ")
        print(calc.do_calculation(input_value, input_value1, input_value2))  
    elif input_opr == 1:
        input_lst1 = input("Please Enter the elements of list 1 seperated by a comma: ").split(',')
        input_lst2 = input("Please Enter the elements of list 2 seperated by a comma: ").split(',')
        print(calc.do_calculation('dp',input_lst1, input_lst2))
