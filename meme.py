from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
        self.rf = RandomForestRegressor()
        self.dataset = [[],[]]

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
        changed = False
        # case : a lot of chromosomes that were XO'd will have None fitness
        # we fill them with prediction from surrogate model first
        if ch[1] is None:
            ch[1] = self.rf.predict([ch[0]])[0]
            changed = True

        while True:
            # has to be written this way or do an explicit deepcopy
            neighbor = [[gene for gene in ch[0]] for _ in range(len(ch[0]))]
            for i in range(len(ch[0])):
                neighbor[i][i] = 1 if neighbor[i][i] == 0 else 0
            pred = self.rf.predict(neighbor)
            argmax, best_neighbor = max(enumerate(pred), key=lambda x:x[1])
            if best_neighbor <= ch[1]:
                # local search unsuccessful
                break
            # local search successful
            changed = True
            for i in range(len(ch[0])):
                ch[0][i] = neighbor[argmax][i]
            ch[1] = best_neighbor
        
        # invalidate the fitness prediction, ask for real evaluation instead
        if changed:
            ch[1] = None

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

        # add only the "UNSEEN" chromosomes into dataset
        # FIXME : is there a better way?
        for ch in pop:
            if ch[0] not in self.dataset[0]:
                self.dataset[0].append(ch[0])
                self.dataset[1].append(ch[1])


    def __train(self):
        # this section only trains the current population, may yield suboptimal model
        # but does not encounter possible explosion in mem usage
        X,Y = list(zip(*self.pop))
        #print(X)
        #print(Y)
        self.rf.fit(X,Y)

        # this tries to fit the entire seen dataset
        #self.rf.fit(*self.dataset)

        rmse = mean_squared_error(self.dataset[1], self.rf.predict(self.dataset[0]), squared=False)
        print(f'rmse = {str(rmse)[:6]}')

    def run(self,gen):
        self.init()
        #self.__train()
        for _ in range(gen):
            pop_a = self.pop[:int(len(self.pop)*self.p_elite)]
            pop_b = self.pop[int(len(self.pop)*self.p_elite):]
            self.__xo(pop_b)
            #for i in range(len(pop_b)):
            #    self.__local(pop_b[i])
            self.pop = pop_a + pop_b
            self.__eval_pop(self.pop)
            percentage = str((Os_SIZE - O0_SIZE +  self.pop[0][1])*100/Os_SIZE)[:4]
            print(f'{self.pop[0][1]} ({percentage}%),{self.pop[-1][1]}')
            #self.__train()


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
            cmd.append('-f'+FLAGS[idx])
        # FIXME : stack-protector cannot take -fno- form, thus causing compile error
        elif 'stack-protector' not in FLAGS[idx]:
            cmd.append('-fno-'+FLAGS[idx])
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
        print(p.stderr.decode('utf-8'))
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
    ga = GA(len(FLAGS),pop_size=1000)
    ga.run(30)


if __name__=='__main__':
    main()
