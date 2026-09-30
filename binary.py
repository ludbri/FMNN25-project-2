
import numpy as np


def digit_to_binary(digit: int, min_bits: int = 4):
    """Convert a base 10 number (Int) to a np.ndarray of 1's and 0's
    5 -> [0,1,0,1] 
    """
    n = digit
    bits = []
    i = 0
    while(n>0):
        q = n//2 # 13//2 => q = 6
        res = n % 2 # 13-2*6 = 1
        bits.append(res) 
        n = q # n = 6
        i += 1 

    while len(bits) < min_bits:
        bits.append(0) 
        
    return np.asarray(bits[::-1])

# ALL THE 4-BIT REPRESENTATION OF NUMBERS 0-9

codes = np.array([digit_to_binary(d) for d in range(10)])

def binary_to_digit(bits: np.array):
    """Take a np.array as input with 4 floats as bits, take the distance to all 
    4-bit codes (0-9) and return the int value of the one with the lowest distance. 
    """
    dists = np.sum((codes - bits) ** 2, axis=1)
    return int(np.argmin(dists))

# TEST
# for d in range(10):
#     assert binary_to_digit(digit_to_binary(d)) == d, d
#     print("Works for {}".format(d))
