
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


valid_binary_seq = np.array([digit_to_binary(d) for d in range(10)])

def closest_digit_from_binary(seq: np.ndarray):
    '''
    Returns the digit which has the smallest euclidian distance between its 
    corresponding binary sequence and seq.

    Parameters
    ----------
    seq : ndarray
        A binary sequence

    Returns
    -------
    int
        The digit for which its binary sequence has the smallest euclidian 
        distance to seq.
    '''
    
    distances = np.linalg.norm(valid_binary_seq - seq, axis=1)
    return int(np.argmin(distances))

# bits = np.asarray([1,0,1,1])
# print("{} to int is {}".format(bits, closest_digit_from_binary(bits)))

# TEST
# for d in range(10):
#     assert binary_to_digit(digit_to_binary(d)) == d, d
#     print("Works for {}".format(d))