#!/usr/bin/env python

"""
    post-test/main_redacted.py
"""

import sys
import numpy as np
assert np.__version__ == '1.16.3', "Must use numpy version 1.16.3"

sys.path.append('dist')
from dist import lib

# --
# Helpers

def check_correctness(check_fn, user_fn):
    """
        Helper function that returns True if user's implementation
        returns the correct response, and False otherwise
    """
    
    if __name__ == "__main__":
        try:
            _ = check_fn(user_fn())
            return True
        except:
            return False

# --
# 3. Create a zero vector of shape (10,)

def user_fn_3():
    # x = ... user code ...
    return x

check_correctness(lib.check_3, user_fn_3)

# --
# 4. Find the size of Z in bytes

def user_fn_4():
    Z = np.arange(128, dtype=np.int32)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_4, user_fn_4)

# --
# 6. Create a zero vector of shape (10, ), then set the 5th element to 1

def user_fn_6():
    # x = ... user code ...
    return x

check_correctness(lib.check_6, user_fn_6)

# --
# 7. Create the vector [10, 11, 12, ..., 47, 48, 49]

def user_fn_7():
    # x = ... user code ...
    return x

check_correctness(lib.check_7, user_fn_7)

# --
# 8. Reverse the vector Z

def user_fn_8():
    np.random.seed(123)
    Z = np.random.randint(0, 100, 100)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_8, user_fn_8)

# --
# 9. Create a 32x32 array w/ value ranging from 0 to 1023
# (elements in the same row should be sequential)

def user_fn_9():
    # x = ... user code ...
    return x

check_correctness(lib.check_9, user_fn_9)

# --
# 10. Find the indices of the nonzeo elements of Z

def user_fn_10():
    np.random.seed(123)
    Z = np.random.choice([0, 1, 2], 100, p=[0.8, 0.1, 0.1])
    
    # x = ... user code ...
    
    assert isinstance(x, np.ndarray)
    return x

check_correctness(lib.check_10, user_fn_10)


# --
# 11. Create the 32x32 identity matrix

def user_fn_11():
    # x = ... user code ...
    return x

check_correctness(lib.check_11, user_fn_11)


# --
# 14. Find the minimum and maximum values of Z

def user_fn_14():
    np.random.seed(123)
    Z = np.random.uniform(0, 1, (10,10))
    
    # min_val = ... user code ...
    # max_val = ... user code ...
    
    return (min_val, max_val)

check_correctness(lib.check_14, user_fn_14)


# --
# 15. Create a 10x10 array w/ 1s on the border and 0s on the interior

def user_fn_15():
    # x = ... user code ...
    return x

check_correctness(lib.check_15, user_fn_15)

# --
# 16. Add a border of zeros around Z

def user_fn_16():
    Z = np.ones((5,5))
    
    # x = ... user code ...
    return x

check_correctness(lib.check_16, user_fn_16)

# --
# Create a 10x10 matrix with values [1, 2, 3, ..., 8, 9] just below the diagonal

def user_fn_18():
    # x = ... user code ...
    return x

check_correctness(lib.check_18, user_fn_18)

# --
# 19. Fill x with a checkerboard pattern (with zeros on the diagonal)

def user_fn_19():
    x = np.zeros((16,16),dtype=int)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_19, user_fn_19)

# --
# 22. Normalize Z by subtracting the mean and dividing by the standard deviation

def user_fn_22():
    np.random.seed(123)
    Z = np.random.uniform(0, 1, 32)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_22, user_fn_22)

# --
# 24. Compute the matrix product of A and B

def user_fn_24():
    np.random.seed(123)
    A = np.random.uniform(0, 1, (5, 3))
    B = np.random.uniform(0, 1, (3, 2))
    
    # x = ... user code ...
    return x

check_correctness(lib.check_24, user_fn_24)


# --
# 25. Negate all elements of x that are between 3 and 8

def user_fn_25():
    x = np.arange(11)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_25, user_fn_25)

# --
# 29. Round Z away from zero

def user_fn_29():
    
    np.random.seed(123)
    Z = np.random.uniform(-10, +10, 10)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_29, user_fn_29)

# --
# 30. Find elements that appear in both a and b

def user_fn_30():
    np.random.seed(123)
    a = np.random.randint(0, 32, 32)
    b = np.random.randint(0, 32, 32)
    
    # x = ... user code ...
    return x

check_correctness(lib.check_30, user_fn_30)

# --
# 37. Create a 16x16 matrix where each row is the vector [0, 1, ..., 14, 15]

def user_fn_37():
    # x = ... user code ...
    return x

check_correctness(lib.check_37, user_fn_37)

# --
# 39. Create a vector of shape (20,) with values evenly spaced between 0 to 1
# Do not include 0 or 1

def user_fn_39():
    # x = ... user code ...
    return x

check_correctness(lib.check_39, user_fn_39)

# --
# 40. Sort Z (ascending order)

def user_fn_40():
    np.random.seed(123)
    Z = np.random.uniform(0, 1, 100)
    # x = ... user code ...
    return x

check_correctness(lib.check_40, user_fn_40)

# --
# 42. Consider two random array A and B.  Write a function to check if they are equal to 
# within numeric tolerance. Assume they are the same shape

def user_fn_42():
    def my_function(A, B):
        # x = ... user code ...
        return x
    
    return my_function

check_correctness(lib.check_42, user_fn_42)

# --
# 43. Make an array immutable (read-only)


def user_fn_43():
    x = np.zeros(10)
    
    # ... user code ...
    
    return x

check_correctness(lib.check_43, user_fn_43)

# --
# 44. Let x and y represent Cartesian coordinates of 32 points.
# Compute vectors r and t such that (r[i], t[i]) is the conversion
# of (x[i], y[i]) to polar coordinates


def user_fn_44():
    np.random.seed(123)
    x = np.random.uniform(0, 1, 32)
    y = np.random.uniform(0, 1, 32)
    
    # r = ... user code ...
    # t = ... user code ...
    
    return (r, t)

check_correctness(lib.check_44, user_fn_44)

# --
# 45. Replace the maximum value of x with -1


def user_fn_45():
    np.random.seed(123)
    x = np.random.uniform(0, 1, 128)
    
    # ... user code ...
    
    return x

check_correctness(lib.check_45, user_fn_45)

# --
# 47. Given vectors X and Y, compute array C such that C[i, j] = 1 / (x[i] - y[j])

def user_fn_47():
    np.random.seed(123)
    x = np.random.uniform(0, 1, 10)
    y = np.random.uniform(0, 1, 20)
    
    # C = ... user code ...
    
    return C

check_correctness(lib.check_47, user_fn_47)

# --
# 48. Print the maximum representable value of numpy's int32 and float32

def user_fn_48():
    # max_int32   = ... user code ...
    # max_float32 = ... user code ...
    
    return (max_int32, max_float32)

check_correctness(lib.check_48, user_fn_48)


# --
# 50. Find the closest value to b in vector a

def user_fn_50():
    np.random.seed(123)
    a = np.arange(100)
    b = np.random.uniform(0, 100)
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_50, user_fn_50)

# --
# 52. Compute the 10x10 Euclidean distance matrix between the rows of Z

def user_fn_52():
    np.random.seed(123)
    Z = np.random.uniform(0, 1, (10, 2))
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_52, user_fn_52)


# --
# 53. Convert Z to an array of np.float32's

def user_fn_53():
    Z = np.arange(10, dtype=np.float64)
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_53, user_fn_53)

# --
# 58. Subtract the mean of each row of Z

def user_fn_58():
    np.random.seed(123)
    Z = np.random.uniform(0, 1, (5, 10))
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_58, user_fn_58)

# --
# 59. Sort Z by it's second columnd (descending order)

def user_fn_59():
    np.random.seed(123)
    Z = np.random.uniform(0, 10, (5, 5))
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_59, user_fn_59)


# --
# 67. Sum over the last two axes of Z

def user_fn_67():
    np.random.seed(123)
    Z = np.random.uniform(0,10, (3,4,3,4))
    
    # x = ... user code ...
    
    return x

check_correctness(lib.check_67, user_fn_67)


# --
# 68. Considering a one-dimensional vector `vals`, compute the means
# of subsets of `vals` using a vector `idx` describing subset indices?
#   That is, x[i] := mean(items of vals where idx == i)


def user_fn_68():
    np.random.seed(123)
    vals = np.random.uniform(0,1,100)
    idx  = np.random.randint(0,10,100)
    
    # ... user code ...
    return x

check_correctness(lib.check_68, user_fn_68)


# --
# 69. Compute the diagonal of the dot product of A and B


def user_fn_69():
    np.random.seed(123)
    A = np.random.uniform(0,1, (5,5))
    B = np.random.uniform(0,1, (5,5))
    
    # x = ... user code ...
    return x

check_correctness(lib.check_69, user_fn_69)


# --
# 70. Add three consecutive zeros between each element of Z


def user_fn_70():
    Z = np.arange(100)
    # ... user code ...
    return x

check_correctness(lib.check_70, user_fn_70)


# --
# 72. Swap the first and second row of x

def user_fn_72():
    x = np.arange(25).reshape(5,5)
    # ... user code ...
    return x

check_correctness(lib.check_72, user_fn_72)


# --
# 74. Create an array x such that np.bincount(x) == C
 
def user_fn_74():
    C = np.array([0, 2, 1, 1, 2, 0, 1])
    # x = ... user code ...
    return x

check_correctness(lib.check_74, user_fn_74)


# --
# 83. Find the most frequent value in Z

def user_fn_83():
    np.random.seed(123)
    Z = np.random.randint(0,10,100)
    # x = ... user code ...
    return x

check_correctness(lib.check_83, user_fn_83)


# --
# 89. Find the 5 largest values of Z

def user_fn_89():
    np.random.seed(123)
    Z = np.random.uniform(0, 100, 1000)
    # x = ... user code ...
    return x

check_correctness(lib.check_89, user_fn_89)


# --
# 94. Extract rows of Z where all elements are not equal
# Eg, extract [0, 0, 1] and [1, 0, 1], but not [0, 0, 0] or [1, 1, 1]

def user_fn_94():
    np.random.seed(123)
    Z = np.random.randint(0, 2, (32,3))
    # x = ... user code ...
    return x

check_correctness(lib.check_94, user_fn_94)


# --
# 96. Extract the unique rows of Z

def user_fn_96():
    np.random.seed(123)
    Z = np.random.randint(0, 2, (128, 3))
    # x =... user code ...
    return x

check_correctness(lib.check_96, user_fn_96)

