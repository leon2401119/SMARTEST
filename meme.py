import sklearn
import os
import random
import subprocess
import glob
from multiprocessing import Pool

CBENCH_PATH = 'cBench_V1.1'
TARGET = 'automotive_bitcount'
FLAGS = None
WORKERS = 4

class GA:
    def __init__(self,ell,pop_size=200, p_xo=0.8, p_elite=0.1):
        self.ell = ell
        self.pop_size = pop_size
        self.p_xo = p_xo
        self.p_elite = p_elite

    def init(self):
        self.pop = [[random.randint(0,1) for _ in range(ell)] for _ in range(self.pop_size)]
        self.fitness = [self.eval(i) for i in self.pop]

    def run(gen):
        for _ in range(gen):
            pass

def retr_gcc_flags():
    FLAGS = []
    with open('flags','r') as f:
        FLAGS.extend(f.read().split('\n'))
    return FLAGS[:-1]


def get_size(bit_vector,ch_num):
    cmd = ['gcc']
    for idx,bit in enumerate(bit_vector):
        if bit:
            cmd.append(FLAGS[idx])
    file_list = glob.glob(os.path.join(CBENCH_PATH,TARGET,'src','*.c'))
    cmd.extend(file_list)
    cmd.extend(['-o',f'{ch_num}'])
    output = subprocess.check_output(cmd).decode('utf-8')
    if len(output):
        print(output)


def main():
    #ga = GA()
    #ga.init()
    global FLAGS
    FLAGS = retr_gcc_flags()
    pool = Pool(processes=WORKERS)
    pool.starmap(get_size, [[[0 for _ in range(len(FLAGS))],i] for i in range(WORKERS)] )


if __name__=='__main__':
    main()
