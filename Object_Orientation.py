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
            for i in range(len(a)):
                dot_sum += (a[i]*b[i])
            return dot_sum
        else:
            print("lists must have equal lenght")
# A. Top-level object of a class Calculator Class
class Calculator:
    def __init__(self):
            self.operations = {'+': Addition(),'-': Subtraction(),'*': Multiplication(),'/': Division(),'%': Modulus(),'dp': DotProduct()
            }

    def do_calculation(self, operation, a, b):
        if operation in self.operations:
            return self.operations[operation].calculate(a, b)
        else:
            return "Operation not supported"

if __name__ == "__main__":
    calc = Calculator()
    print(calc.do_calculation('+', 15, 3))  
    print(calc.do_calculation('-', 15, 3))  
    print(calc.do_calculation('*', 15, 3))  
    print(calc.do_calculation('/', 15, 3)) 
    print(calc.do_calculation('%', 15, 3))
    print(calc.do_calculation('dp', [1,3,2], [4,1,3])) 
