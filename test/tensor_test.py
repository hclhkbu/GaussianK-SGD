import numpy as np

P = 32
k = 10
flags = np.zeros(k, dtype=np.uint32)
flags2 = np.zeros(k, dtype=np.uint32)
idx = 4
rank = 30
print('before: {:b}'.format(flags[idx]))
def set_bit(value, bit):
    return value | (1<<bit)

def check_exist(value, rank):
    bit = (value >> rank) & 1
    return bit

flags[idx]=set_bit(flags[idx], rank)
print('after: {:b}\n'.format(flags[idx]))

for f in flags:
    #print('{:b}'.format(f))
    print('{:b}'.format(check_exist(f, rank)))




