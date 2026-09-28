
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


# print("5 to bin is {}".format(digit_to_binary(5)))

def binary_to_digit(bits: np.array):
    """Convert an array of ones and zeroes anc convert to an int
    [0,1,0,1] -> 5
    """
    exp = len(bits) - 1
    res = 0
    for bit in bits:
        if bit == 1:
            res += 2**exp
        exp-=1

    return res

# bits = np.asarray([1,0,1,1])
# print("{} to int is {}".format(bits, binary_to_digit(bits)))