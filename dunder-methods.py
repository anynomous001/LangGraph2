#!/usr/bin/env python3
"""
Complete Guide to Python Dunder Methods (Magic Methods)
Demonstrates the most common and useful dunder methods
"""

class Person:
    """Example class demonstrating various dunder methods"""
    
    def __init__(self, name, age):
        """Constructor - called when creating new instance"""
        self.name = name
        self.age = age
        print(f"__init__ called: Creating {name}")
    
    def __str__(self):
        """String representation for humans (print, str())"""
        return f"Person(name='{self.name}', age={self.age})"
    
    def __repr__(self):
        """String representation for developers (repr(), debugging)"""
        return f"Person('{self.name}', {self.age})"
    
    def __len__(self):
        """Called when using len() function"""
        return len(self.name)
    
    def __eq__(self, other):
        """Equality comparison (==)"""
        if isinstance(other, Person):
            return self.name == other.name and self.age == other.age
        return False
    
    def __lt__(self, other):
        """Less than comparison (<)"""
        if isinstance(other, Person):
            return self.age < other.age
        return NotImplemented
    
    def __le__(self, other):
        """Less than or equal (<=)"""
        return self < other or self == other
    
    def __gt__(self, other):
        """Greater than (>)"""
        if isinstance(other, Person):
            return self.age > other.age
        return NotImplemented
    
    def __ge__(self, other):
        """Greater than or equal (>=)"""
        return self > other or self == other
    
    def __add__(self, other):
        """Addition operator (+)"""
        if isinstance(other, Person):
            # Create a "family" when adding two people
            return f"{self.name} & {other.name} Family"
        return NotImplemented
    
    def __sub__(self, other):
        """Subtraction operator (-)"""
        if isinstance(other, Person):
            return abs(self.age - other.age)
        return NotImplemented
    
    def __mul__(self, other):
        """Multiplication operator (*)"""
        if isinstance(other, int):
            # Repeat name multiple times
            return self.name * other
        return NotImplemented
    
    def __getitem__(self, key):
        """Index access - person[key]"""
        if key == 0:
            return self.name
        elif key == 1:
            return self.age
        else:
            raise IndexError("Person index out of range")
    
    def __setitem__(self, key, value):
        """Index assignment - person[key] = value"""
        if key == 0:
            self.name = value
        elif key == 1:
            self.age = value
        else:
            raise IndexError("Person index out of range")
    
    def __contains__(self, item):
        """'in' operator - item in person"""
        return item.lower() in self.name.lower()
    
    def __call__(self, greeting="Hello"):
        """Make object callable like a function"""
        return f"{greeting}, I'm {self.name}!"
    
    def __hash__(self):
        """Make object hashable (for sets, dict keys)"""
        return hash((self.name, self.age))
    
    def __bool__(self):
        """Truth value testing"""
        return self.age > 0  # Person is "truthy" if age > 0
    
    def __del__(self):
        """Destructor - called when object is destroyed"""
        print(f"__del__ called: {self.name} is being destroyed")


class MathVector:
    """Example class for mathematical operations"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        """Vector addition"""
        if isinstance(other, MathVector):
            return MathVector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __sub__(self, other):
        """Vector subtraction"""
        if isinstance(other, MathVector):
            return MathVector(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        """Scalar multiplication"""
        if isinstance(scalar, (int, float)):
            return MathVector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Reverse multiplication (scalar * vector)"""
        return self * scalar
    
    def __abs__(self):
        """Absolute value (magnitude)"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __neg__(self):
        """Unary minus (-vector)"""
        return MathVector(-self.x, -self.y)


class SmartList:
    """Custom list-like class demonstrating container methods"""
    
    def __init__(self):
        self._items = []
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, value):
        self._items[index] = value
    
    def __delitem__(self, index):
        del self._items[index]
    
    def __iter__(self):
        """Make object iterable"""
        return iter(self._items)
    
    def __reversed__(self):
        """Support for reversed()"""
        return reversed(self._items)
    
    def __contains__(self, item):
        return item in self._items
    
    def append(self, item):
        self._items.append(item)
    
    def __str__(self):
        return f"SmartList({self._items})"


class ContextManager:
    """Example of context manager dunder methods"""
    
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        """Called when entering 'with' block"""
        print(f"Entering context: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block"""
        print(f"Exiting context: {self.name}")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # Don't suppress exceptions


def demonstrate_dunder_methods():
    """Demonstrate various dunder methods in action"""
    
    print("=== BASIC DUNDER METHODS ===")
    
    # Constructor and string representations
    person1 = Person("Alice", 30)
    person2 = Person("Bob", 25)
    
    print(f"str(person1): {str(person1)}")
    print(f"repr(person1): {repr(person1)}")
    print(f"len(person1): {len(person1)}")
    
    print("\n=== COMPARISON OPERATORS ===")
    print(f"person1 == person2: {person1 == person2}")
    print(f"person1 > person2: {person1 > person2}")
    print(f"person1 < person2: {person1 < person2}")
    
    print("\n=== ARITHMETIC OPERATORS ===")
    print(f"person1 + person2: {person1 + person2}")
    print(f"person1 - person2: {person1 - person2}")
    print(f"person1 * 3: {person1 * 3}")
    
    print("\n=== INDEXING AND CONTAINMENT ===")
    print(f"person1[0]: {person1[0]}")
    print(f"person1[1]: {person1[1]}")
    print(f"'Ali' in person1: {'Ali' in person1}")
    print(f"'xyz' in person1: {'xyz' in person1}")
    
    # Modify through indexing
    person1[1] = 31
    print(f"After person1[1] = 31: {person1}")
    
    print("\n=== CALLABLE OBJECTS ===")
    print(f"person1(): {person1()}")
    print(f"person1('Hi'): {person1('Hi')}")
    
    print("\n=== HASH AND BOOL ===")
    print(f"hash(person1): {hash(person1)}")
    print(f"bool(person1): {bool(person1)}")
    
    # Create person with age 0
    baby = Person("Baby", 0)
    print(f"bool(baby): {bool(baby)}")
    
    print("\n=== MATH VECTOR EXAMPLE ===")
    v1 = MathVector(3, 4)
    v2 = MathVector(1, 2)
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 + v2: {v1 + v2}")
    print(f"v1 - v2: {v1 - v2}")
    print(f"v1 * 2: {v1 * 2}")
    print(f"3 * v1: {3 * v1}")
    print(f"abs(v1): {abs(v1)}")
    print(f"-v1: {-v1}")
    
    print("\n=== SMART LIST EXAMPLE ===")
    smart_list = SmartList()
    smart_list.append(1)
    smart_list.append(2)
    smart_list.append(3)
    
    print(f"smart_list: {smart_list}")
    print(f"len(smart_list): {len(smart_list)}")
    print(f"smart_list[1]: {smart_list[1]}")
    print(f"2 in smart_list: {2 in smart_list}")
    
    print("Iterating:")
    for item in smart_list:
        print(f"  {item}")
    
    print("Reversed:")
    for item in reversed(smart_list):
        print(f"  {item}")
    
    print("\n=== CONTEXT MANAGER EXAMPLE ===")
    with ContextManager("test_context"):
        print("Inside with block")
        # Uncomment next line to see exception handling
        # raise ValueError("Test exception")
    
    print("\n=== CLEANUP ===")
    # Objects will be destroyed when they go out of scope


if __name__ == "__main__":
    demonstrate_dunder_methods()
    print("Program finished - watch for __del__ messages")