import sklearn
import os
import random
import subprocess
import glob
from multiprocessing import Pool
import copy


CBENCH_PATH = 'cBench_V1.1'
TARGET = 'automotive_bitcount'
TMP_DIR = 'tmpfiles'
FLAGS = None
WORKERS = 4
O0_SIZE = None
Os_SIZE = None

class GA:
    def __init__(self,ell,pop_size=200, p_xo=0.8, p_elite=0.1):
        self.ell = ell
        self.pop_size = pop_size
        self.p_xo = p_xo
        self.p_elite = p_elite

    def init(self):
        # chromosome is formatted as [[genes],fitness], fitness is None when
        # the chromosomenot is not evaulated yet
        self.pop = [[[random.randint(0,1) for _ in range(self.ell)],None] for _ in range(self.pop_size)]
        self.__eval_pop(self.pop)

    def __xo(self,pop):
        # roulette-wheel
        # FIXME : what about negative value??
        # prepare the base for rw selection
        # chromosomes that cause compile error will not be preserved at all
        adjusted_pop = [ch for ch in pop if ch[1] is not None]
        adjust_value = 0
        if adjusted_pop[-1][1] < 0:
            adjust_value = adjusted_pop[-1][1]
            for ch in adjusted_pop:
                ch[1] -= adjust_value # pop is sorted by fitness already
        
        total_fitness = 0
        for ch in adjusted_pop:
            total_fitness += ch[1]

        new_pop = []

        while len(new_pop) < len(pop):
            tmp1 = random.randint(1,total_fitness)
            tmp2 = random.randint(1,total_fitness)
            accum,i = 0,-1
            while accum < tmp1:
                i += 1
                accum += adjusted_pop[i][1]
            parent_a = copy.deepcopy(adjusted_pop[i])
            accum,i = 0,-1
            while accum < tmp2:
                i += 1
                accum += adjusted_pop[i][1]
            parent_b = copy.deepcopy(adjusted_pop[i])

            if random.random() < self.p_xo: # do crossover (one-point for now)
                site = random.randint(1,len(parent_a[0])-2)
                parent_a[0][site:], parent_b[0][:site] = parent_b[0][site:], parent_a[0][:site]
                parent_a[1], parent_b[1] = None, None
            
            # revert from the skew made for rw-selection
            if parent_a[1] is not None:
                parent_a[1] += adjust_value
            if parent_b[1] is not None:
                parent_b[1] += adjust_value

            if len(new_pop) == len(pop)-1:  # correction when pop has odd number of chromosomes
                new_pop.append(parent_a)

            else:
                new_pop.extend([parent_a,parent_b])

        # assigning pop = new_pop here is not going to change pop for the caller, which is not what we intended
        for i in range(len(new_pop)):
            pop[i] = new_pop[i]


    def __local(self,ch):
        changed = True
        while changed:
            changed = False
            for i in range(len(ch)):
                ch[i] = 0 if ch[i] == 1 else 0
                # TODO

    def __eval_pop(self,pop):
        # evaluate whole population and sort from fitness high to low
        pool = Pool(processes=WORKERS)
        workload = [[ch,i] for i,ch in enumerate(pop) if ch[1] is None]
        #print(len(workload))
        r = pool.starmap(get_fitness, workload)
        for i in range(len(workload)):
            pop[workload[i][1]][1] = r[i]
        
        # testing purpose for single process
        #for i,ch in enumerate(pop):
        #    if ch[1] == None:
        #        ch[1] = get_fitness(ch,i)

        #pop = list(zip(list(zip(*pop))[0],r))
        # tmp is used to sort None value to the back
        tmp = min([num for num in pop if num is not None])
        pop.sort(key=lambda x:x[1] if x[1] is not None else tmp-1, reverse=True)

    def run(self,gen):
        self.init()
        for _ in range(gen):
            pop_a = self.pop[:int(len(self.pop)*self.p_elite)]
            pop_b = self.pop[int(len(self.pop)*self.p_elite):]
            self.__xo(pop_b)
            self.pop = pop_a + pop_b
            self.__eval_pop(self.pop)
            #print(self.pop)
            print(self.pop[0][1],self.pop[-1][1])


def retr_gcc_flags():
    FLAGS = []
    with open('flags','r') as f:
        FLAGS.extend(f.read().split('\n'))
    return FLAGS[:-1]

def get_fitness(ch,ch_num):
    bit_vector = ch[0]
    cmd = ['gcc']
    for idx,bit in enumerate(bit_vector):
        if bit:
            cmd.append(FLAGS[idx])
    file_list = glob.glob(os.path.join(CBENCH_PATH,TARGET,'src','*.c'))
    cmd.extend(file_list)
    cmd.extend(['-o',f'{TMP_DIR}/{ch_num}'])
    if not compile(cmd):
        return None

    cmd = ['size',f'{TMP_DIR}/{ch_num}']
    size = get_size(cmd)
    return O0_SIZE - size

def compile(cmd):
    p = subprocess.run(cmd,capture_output=True)
    # in gcc, warnings and errors are clobbered in stderr for some reason
    # FIXME : is it possible to seperate them?
    if p.returncode:    # indicates compile err
        print(p.returncode,len(p.stdout),len(p.stderr))
        return False
    #output = subprocess.check_output(cmd).decode('utf-8')
    return True

def get_size(cmd):
    output = subprocess.check_output(cmd).decode('utf-8')
    # stats = [text,data,bss]
    stats = [int(i) for i in output.split('\n')[1].split('\t')[:3] if len(i)]
    return stats[0] + stats[1]

def get_baseline_size():
    cmd = ['gcc','-O0']
    file_list = glob.glob(os.path.join(CBENCH_PATH,TARGET,'src','*.c'))
    cmd.extend(file_list)
    cmd.extend(['-o',f'{TMP_DIR}/O0'])
    if not compile(cmd):
        return False
    cmd[1],cmd[-1] = '-Os',f'{TMP_DIR}/Os'
    if not compile(cmd):
        return False
    cmd = ['size',f'{TMP_DIR}/O0']
    global O0_SIZE
    O0_SIZE = get_size(cmd)
    cmd[1] = f'{TMP_DIR}/Os'
    global Os_SIZE
    Os_SIZE = get_size(cmd)
    return True

def main():
    global FLAGS
    FLAGS = retr_gcc_flags()
    get_baseline_size()
    print('Os Fitness : ' + str(O0_SIZE - Os_SIZE))
    ga = GA(len(FLAGS),pop_size=200)
    ga.run(100)


if __name__=='__main__':
    main()
