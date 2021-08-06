from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import gmean
from scipy import mean
import os
import random
import subprocess
import glob
from multiprocessing import Pool
import copy
import matplotlib.pyplot as plt

# Settings & Tunables
BASE_IS_OS = True
CBENCH_PATH = 'cBench_V1.1'
TARGET = None
TMP_DIR = 'tmpfiles'
FIG_DIR = 'results/fig'
BIN_DIR = 'results/bin'
WORKERS = 6
ADVANCED_STATS = True

# Global Variables
FLAGS = None
O0_SIZE = None
Os_SIZE = None

class GA:
    def __init__(self,ell,pop_size=200, p_xo=0.8, p_elite=0.1,meme=True):
        self.ell = ell
        self.pop_size = pop_size
        self.p_xo = p_xo
        self.p_elite = p_elite
        self.meme = meme
        self.rf = RandomForestRegressor()
        self.dataset = [[],[]]

    def __init(self):
        if not get_baseline_size():
            return False

        print('Local Search : ',end='')
        if self.meme:
            print('ON')
        else:
            print('OFF')
        # chromosome is formatted as [[genes],fitness], fitness is None when
        # the chromosomenot is not evaulated yet
        self.pop = [[[random.randint(0,1) for _ in range(self.ell)],None] for _ in range(self.pop_size)]
        self.__eval_pop(self.pop)
        self.statistics = []    # [[gen1_stats],[gen2_stats],...]
        self.adv_statistics = []
        return True

    def __record_stats(self):
        # stat = [max_fit,mean_fit,min_fit,diversity_gmean,diversity_mean,diversity_worst,diversity_best]
        # we have to do some little filtering here becuase some chromosomes may have None fitness
        # even after eval phase simply because it won't compile somehow

        filtered_fitness = [ch[1] for ch in self.pop if ch[1] is not None]

        #mean_fitness = mean(filtered_fitness)
        mean_fitness = gmean(filtered_fitness)
        self.statistics.append([filtered_fitness[0],mean_fitness,filtered_fitness[-1]] + self.report_diversity())

    def report_diversity(self,stdout=False):
        p_vec = [0 for _ in range(self.ell)]
        for ch in self.pop:
            for i,gene in enumerate(ch[0]):
                p_vec[i] += gene
        for i in range(self.ell):
            p_vec[i] /= self.pop_size
            p_vec[i] = (0.5 - abs(p_vec[i]-0.5))*200    # normalize 0.0 ~ 0.5 to 0 ~ 100, also 1.0 ~ 0.5

        #worst = 1-max(p_vec) if 1-max(p_vec) < min(p_vec) else min(p_vec)
        #best = 1
        #for p in p_vec:
        #    if abs(p-0.5) < abs(best-0.5):
        #        best = p

        best,worst = max(p_vec),min(p_vec)

        if stdout:
            print('------- population diversity report -------')
            print("Geomean = {:.3f}".format(gmean(p_vec)))
            print("Mean = {:.3f}".format(mean(p_vec)))
            print("Worst = {:.3f}".format(worst))
            print("Best = {:.3f}".format(best))
            print('-------------------------------------------')

        return [gmean(p_vec),mean(p_vec),worst,best]

    def __plot(self):
        x = list(range(len(self.statistics)))
        max_fit,mean_fit,min_fit,gmean,mean,worst,best = list(zip(*self.statistics))

        ax1 = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=2)
        plt.ylabel('Fitness')

        plt.plot(x,max_fit,label='Max')
        plt.plot(x,mean_fit,label='Mean')
        #plt.plot(x,min_fit,label='Min')
        if self.meme and ADVANCED_STATS:
            plt.plot(x[1:],self.adv_statistics,label='Est')  # we don't have the data in init phase (gen 0) yet
        plt.axhline(O0_SIZE - Os_SIZE, color='black',linestyle = '--',label='Os')
        plt.legend(loc = 'lower right')

        ax2 = plt.subplot2grid((3,3), (2,0), colspan=3, rowspan=1)
        plt.xlabel('Generation')
        plt.ylabel('Gene Diversity')
        #plt.plot(x,gmean,label='GMean')
        plt.plot(x,mean,label='Mean')
        plt.plot(x,best,label='Best')
        plt.plot(x,worst,label='Worst')
        plt.legend(loc = 'lower right')

        plt.tight_layout()
        #plt.show()
        gen = len(self.statistics)-1
        plt.savefig(f'{FIG_DIR}/{TARGET}_{self.meme}_{BASE_IS_OS}_{gen}_{self.pop_size}_{self.p_xo}_{self.p_elite}.png')

    def __xo(self,parent_a,parent_b):
        self.__n_point_xo(parent_a,parent_b)

    def __n_point_xo(self,parent_a,parent_b,n=1):
        candidate_sites = [i for i in range(1,len(parent_a[0])-1)]
        sites = random.sample(candidate_sites, n)
        sites = [None] + sorted(sites) + [None]
        parent_switch = False
        for t in range(len(sites)-1):
            if parent_switch:
                parent_a[0][sites[t]:sites[t+1]], parent_b[0][sites[t]:sites[t+1]] = parent_b[0][sites[t]:sites[t+1]], parent_a[0][sites[t]:sites[t+1]]
            parent_switch = not parent_switch

        parent_a[1], parent_b[1] = None, None

    def __selection(self,pop=None): # cannot do pop=self.pop, why??
        if not pop:
            pop = self.pop
        self.__rw_selection(pop)

    def __rw_selection(self,pop):
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
                self.__xo(parent_a,parent_b)

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
        
        est_fitness = ch[1]

        # invalidate the fitness prediction, ask for real evaluation instead
        if changed:
            ch[1] = None

        # returns the estimated fitness (real or predicted)
        return est_fitness

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
        tmp = min([num[1] for num in pop if num[1] is not None])
        #print(tmp)
        pop.sort(key=lambda x:x[1] if x[1] is not None else tmp-1, reverse=True)

        # add only the "UNSEEN" chromosomes into dataset
        # FIXME : is there a better way?
        for ch in pop:
            if ch[0] not in self.dataset[0] and ch[1] is not None:
                # some benchmarks (e.g. tiff2rgba) will occasionally get compile err
                # for specific chromosomes, so filter it out here
                # turns out it is cBench_V1.1/consumer_tiff2rgba/src/tif_fax3.c:1507:1: internal compiler error: Segmentation fault
                # WTF is this shit?
                self.dataset[0].append(ch[0])
                self.dataset[1].append(ch[1])


    def __train(self):
        # this section only trains the current population, may yield suboptimal model
        # but does not encounter possible explosion in mem usage
        #X,Y = list(zip(*self.pop))
        #print(X)
        #print(Y)
        #self.rf.fit(X,Y)

        # this tries to fit the entire seen dataset
        self.rf.fit(*self.dataset)

        rmse = mean_squared_error(self.dataset[1], self.rf.predict(self.dataset[0]), squared=False)
        print(f'rmse = {str(rmse)[:6]}')

    def run(self,gen):
        print(f'Optimizing "{TARGET}"...')

        if not self.__init():
            print(f'Fail to optimize "{TARGET}"')
            print('\n')
            return False
       
        self.__record_stats()
        #self.report_diversity()
        if self.meme:
            self.__train()
        for g in range(gen):
            pop_a = self.pop[:int(len(self.pop)*self.p_elite)]
            pop_b = self.pop[int(len(self.pop)*self.p_elite):]
            self.__selection(pop_b)
            if self.meme:
                est_record = []
                for i in range(len(pop_b)):
                    est_record.append(self.__local(pop_b[i]))
                    
                if ADVANCED_STATS:
                    self.adv_statistics.append(mean(est_record))

            self.pop = pop_a + pop_b
            self.__eval_pop(self.pop)
            percentage = (Os_SIZE - O0_SIZE +  self.pop[0][1])*100/Os_SIZE
            self.__record_stats()
            print("Gen_{} : {} ({:.2f}%),{}".format(g,self.pop[0][1],percentage,self.pop[-1][1]))
            #self.report_diversity()
            if self.meme:
                self.__train()

        # save the optimized binary & compile options
        opt_cmd = get_cmd(self.pop[0][0],f'../{BIN_DIR}/{TARGET}_{self.meme}_{BASE_IS_OS}_{gen}_{self.pop_size}_{self.p_xo}_{self.p_elite}')
        compile(opt_cmd)
        with open(f'{BIN_DIR}/{TARGET}_{self.meme}_{BASE_IS_OS}_{gen}_{self.pop_size}_{self.p_xo}_{self.p_elite}.txt','w') as f:
            for token in opt_cmd:
                print(token,end=' ',file=f)
        
        self.__plot()
        print('\n')
        return True


def retr_gcc_flags():
    FLAGS = []
    with open('flags','r') as f:
        FLAGS.extend(f.read().split('\n'))
    return FLAGS[:-1]

def get_fitness(ch,ch_num):
    cmd = get_cmd(ch[0],ch_num)
    if not compile(cmd):
        return None

    cmd = ['size',f'{TMP_DIR}/{ch_num}']
    size = get_size(cmd)
    return O0_SIZE - size

def get_cmd(bit_vector,ch_num): # ch_num is used to avoid race condition on File System
    cmd = ['gcc']
    if BASE_IS_OS:
        cmd += ['-Os']
    for idx,bit in enumerate(bit_vector):
        if bit:
            cmd.append('-f'+FLAGS[idx])
        # FIXME : stack-protector cannot take -fno- form, thus causing compile error
        elif 'stack-protector' not in FLAGS[idx]:
            cmd.append('-fno-'+FLAGS[idx])
    file_list = glob.glob(os.path.join(CBENCH_PATH,TARGET,'src','*.c'))
    cmd.extend(file_list)
    cmd.extend(['-o',f'{TMP_DIR}/{ch_num}'])
    return cmd

def compile(cmd):
    p = subprocess.run(cmd,capture_output=True)
    # in gcc, warnings and errors are clobbered in stderr for some reason
    # FIXME : is it possible to seperate them?
    if p.returncode:    # indicates compile err
        #print(p.returncode,len(p.stdout),len(p.stderr))
        # second chance to also link with math libraries
        if subprocess.run(cmd+['-lm'],capture_output=True).returncode:
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
    print('Os Fitness : ' + str(O0_SIZE - Os_SIZE))
    return True




def main():
    global FLAGS
    FLAGS = retr_gcc_flags()
    
    global TARGET
    f = glob.glob(os.path.join(CBENCH_PATH,'*','src/')) # get all folders under CBENCH_PATH
    for d in f:
        TARGET = os.path.basename(d[:-5]) # dispose of '/src/' to get the real folders

        #TARGET = 'office_rsynth'
        TARGET = 'consumer_jpeg_c'

        # FIXME : about 15 of total 32 benchmarks will fail to compile(link) because of some library issues
        ga = GA(len(FLAGS),pop_size=50,meme=False)
        ga.run(5)

        return


if __name__=='__main__':
    main()
