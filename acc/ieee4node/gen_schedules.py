
import numpy as np

#file_name = 'sample'
file_name = 'load'
start_time = '2000-01-01 00:00:00 EST'
base_value = 1800000.000+871779.789j
#sample_size = 10
sample_size = 10000
interval = '+60s'
seed = 1

np.random.seed(seed)
scale_changes = np.random.randint(70, 101, sample_size) / 100

f = open('{}.{}'.format(file_name, 'player'), 'w')
symbol = '+' if base_value.imag >=0  else '-'
f.write('{}, {:.3f}{}{:.3f}j'.format(start_time, base_value.real, symbol, abs(base_value.imag)))
for loop0 in range(0, sample_size):
    new_value = base_value.real * scale_changes[loop0] + (base_value - base_value.real)
    print(loop0, scale_changes[loop0], new_value)
    f.write('\n{}, {:.3f}{}{:.3f}j'.format(interval, new_value.real, symbol, abs(new_value.imag)))
f.close()

