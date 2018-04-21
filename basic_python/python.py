from basic_python.basictools import pause


def get_y(a, b):
    return lambda x: a * x + b


y1 = get_y(14, 3)
print(y1(2))  # 结果为2

print((lambda x, y: x * 3 + y * 78)(90, 5))


def get_y_normal(a, b):
    def func(x):
        return a * x + b

    return func


y2 = get_y_normal(12, 3)
print(y2)
print(y2(4))
pause()
'''
Create a function.
'''


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


print(quicksort([3, 6, 8, 10, 1, 2, 1]))

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

'''
Create a class.
'''


class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)


g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()  # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)  # Call an instance method; prints "HELLO, FRED!"
