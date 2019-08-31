"""
Generate dropout binary for each layer multiple times.
"""
import numpy as np

def write_descriptor(fname):


    g2 = []
    with open(fname, 'r') as fin:
        for line in fin:
            finish = False
            if 'g2    8    2' in line:
                i = 0
                for line in fin:
                    line = line[:line.index('#')]
                    line = [float(x) for x in line.split()]
                    g2.append(line)
                    i += 1
                    if i == 8:
                        finish = True
                        break
            if finish:
                break

    g4 = []
    with open(fname, 'r') as fin:
        for line in fin:
            finish = False
            if 'g4    43    3' in line:
                i = 0
                for line in fin:
                    line = line[:line.index('#')]
                    line = [float(x) for x in line.split()]
                    g4.append(line)
                    i += 1
                    if i == 43:
                        finish = True
                        break
            if finish:
                break

    mean = []
    with open(fname, 'r') as fin:
        for line in fin:
            finish = False
            if '# mean' in line:
                i = 0
                for line in fin:
                    mean.append(float(line))
                    i += 1
                    if i == 51:
                        finish = True
                        break
            if finish:
                break


    std = []
    with open(fname, 'r') as fin:
        for line in fin:
            finish = False
            if '# standard derivation' in line:
                i = 0
                for line in fin:
                    std.append(float(line))
                    i += 1
                    if i == 51:
                        finish = True
                        break
            if finish:
                break


    with open('descriptor.params', 'w') as fout:
        fout.write('cos  # cutoff type\n\n')
        fout.write('1  # number of species\n\n')
        fout.write('# species 1    species 2    cutoff\nC  C  5\n\n')
        fout.write('#' + '='*80 + '\n')
        fout.write('# symmetry functions\n')
        fout.write('#' + '='*80 + '\n\n')
        fout.write('2    # number of symmetry function types\n\n')

        fout.write('#sym_function    rows    cols\n\n')
        fout.write('g2  8  2\n')
        fout.write('# eta  Rs\n')
        for line in g2:
            fout.write('{:23.15e} {:2d}\n'.format(line[0], int(line[1])))
        fout.write('\ng4  43  3\n')
        fout.write('# zeta  lambda  eta\n')
        for line in g4:
            fout.write('{:2d} {:2d} {:23.15e}\n'.format(int(line[0]), int(line[1]), line[2]))


        fout.write('\ncenter_and_normalize  True\n51  # descriptors size\n\n')
        fout.write('# mean\n')
        for line in mean:
            fout.write('{:23.15e}\n'.format(line))
        fout.write('# standard deviation\n')
        for line in std:
            fout.write('{:23.15e}\n'.format(line))



def write_nn_params(fname):


    with open(fname, 'r') as fin, open('NN.params', 'w') as fout:
        for line in fin:
            finish = False
            if '# ANN structure and parameters' in line:
                fout.write('#' + '='*80 + '\n')
                fout.write(line)
                for line in fin:
                    fout.write(line)





def get_num_units_and_keep_probability(fname):
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
  fname = 'ann_kim_step2960.params_ivdw202_nlayer4_n128_kr0.9-varying_alat_set_3_percent'
  num_units, keep_prob = get_num_units_and_keep_probability(fname)
  write_dropout_binary(num_units, keep_prob, repeat=100)
  write_descriptor(fname)
  write_nn_params(fname)
