"""
Generate dropout binary for each layer multiple times.
"""
import numpy as np

def get_num_units_and_keep_probability(fname='ANN.params'):
  num_units = []
  with open(fname, 'r') as fin:
    for line in fin:
      if 'number of descriptors' in line:
        n = int(line.strip().split()[0])
        num_units.append(n)
      if 'size of each layer' in line:
        line = line[:line.index('#')]
        line = [int(i) for i in line.strip().split()]
        num_units.extend(line)
      if 'keep probability' in line:
        line = line[:line.index('#')]
        keep_prob = [float(i) for i in line.strip().split()]

  return num_units, keep_prob


def write_dropout_binary(num_units, keep_prob, repeat=50, fname='dropout_binary.params'):
  with open(fname, 'w') as fout:
    fout.write('{} # number of repeat\n'.format(repeat))
    for rep in range(repeat):
      fout.write('#'+ '='*80 + '\n')
      fout.write('# instance {}\n'.format(rep))
      for i in range(len(keep_prob)):
        fout.write('# layer {}\n'.format(i))
        n = num_units[i]
        k = keep_prob[i]
        rnd = np.floor(np.random.uniform(k, k+1, n))
        rnd = np.asarray(rnd, dtype=np.intc)
        for d in rnd:
          d = 1 if d > 1 else d
          d = 0 if d < 0 else d
          fout.write('{} '.format(d))
        fout.write('\n')


if __name__ == '__main__':
  np.random.seed(305)
  num_units, keep_prob = get_num_units_and_keep_probability(fname='ANN.params')
  write_dropout_binary(num_units, keep_prob, repeat=50)

