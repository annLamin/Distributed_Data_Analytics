# 1. Calculator:
print("Question 1.A")
# The five functions perform basc maths operations. 
def addition(a,b):
    add_numbers = a + b
    return add_numbers
addition(4,2)
print(f'the summation of the input values is: {addition(4,2)}')

def subtraction(a,b):
    sub_numbers = a - b
    return sub_numbers

print(f'the subtraction of the input values is: {subtraction(4,2)}')

def multiplication(a,b):
    multiply = a * b
    return multiply
print(f'the multiplication of the input values is: {multiplication(4,2)}')

def division(a,b):
    devide_numbers = a / b
    return devide_numbers
print(f'the division of the input values is: {division(4,2)}')


def modulus(a,b):
    modulus_numbers = a % b
    return modulus_numbers

print(f'the modulos of the input values is: {modulus(4,2)}')


print("Question 1.B")

def dot_products(a,b):
    dot_sum = 0
    for i in range(len(a)):
        dot_sum += (a[i]*b[i])
    return dot_sum

print(f'the dot product of the two list is: {dot_products([1,2,3,4,5],[4,5,6,7,8])}')



print("Question 1.C: Input prompt and Control mechanism")
def input_prompt():
    input_opr = int(input("Which operation do you want to perform? 0 = basic operations 1 = dot produc: "))
    if input_opr == 0: 
        input_value1 = int(input("Please input your first degit: "))
        input_value2 = int(input("Please input your second degit: "))
        input_value = input("Select one of the following Operations you want to perform (+,-,*,/,%): ")
        if input_value == "+":
            print(addition(input_value1,input_value2))
        elif input_value == "-":
            print(subtraction(input_value1,input_value2))
        elif input_value == "*":
            print(multiplication(input_value1,input_value2))
        elif input_value == "/":
            print(division(input_value1,input_value2))
        elif input_value == "%":
            print(modulus(input_value1,input_value2))
        else:
            print("Invalid Operator")
    elif input_opr == 1:
        input_lst1 = input("Please Enter the elements of list 1 seperated by a comma: ").split(',')
        input_lst2 = input("Please Enter the elements of list 2 seperated by a comma: ").split(',')
        if len(input_lst1) == len(input_lst2):
            input_lst1 = [int(i) for i in input_lst1]
            input_lst2 = [int(i) for i in input_lst2]
            print(dot_products(input_lst1,input_lst2))
        else:
            print("Length of list most be equal")
    else:
        print(f"There is currently no operation for {input_opr}: Use 0 or 1")
input_prompt()