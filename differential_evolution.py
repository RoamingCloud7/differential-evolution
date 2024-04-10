import time
import numpy as np

# Create th  Differential Evolution class
class DifferentialEvolution():
    # The initialization function
    # func: the function to calculate the corresponding value of the individual
    # n_dim:    the dimension of the individual
    # F:        Mutation scaling factor
    # size_pop: the size of the population
    # max_iter: the maximum number of iterations
    # prob_mut: the probability of crossover
    # lb:       lower limits of every dimension of the individual
    # ub:       upper limits of every dimension of the individual
    def __init__(self, func, n_dim, F=0.5, size_pop=50, max_iter=200, prob_mut=0.5,lb=-1, ub=1):
        # generation_best_X/Y：The best individual and its objective value after several iterations
        self.generation_best_X = [-99]
        self.generation_best_Y = [-99]
        self.all_history_Y=[]

        self.func= self.func_transformer(func)  
        self.F = F
        self.n_dim=n_dim
        self.size_pop=size_pop
        self.max_iter=max_iter
        self.prob_mut=prob_mut
        # X: the initial population
        # V: the population after mutation
        # U: the population after crossover
        # Y: the objective scores of every individuals
        self.X,self.V, self.U= None, None,None
        self.Y=[]
        self.convergeTime=500
        self.totalTime=0
        # create the boundary of the population
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)

    # Transform the input function
    # The func function can only calculate the corresponding objective values for a single set of plant parameters
    # while the converted function can calculate objective values for multiple sets of crop parameters.
    def func_transformer(self,func):

        def func_transformed(X):
            return np.array([func(x) for x in X])

        return func_transformed

    # initialize the population
    def crtbp(self):
        # X: the set of population
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))

    # Mutation process
    def mutation(self):
        X = self.X
        # select the individuals for mutation
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        # V: the population after mutation
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])
        # create a random population for back-up replacement
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        # If the partial component of some individuals in the population after the mutation is out of bounds
        # it will be replaced by the corresponding value of back-up randomized population
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)

    # Crossover process
    def crossover(self):
        # mask为一个bool型数组，为true的值表示种群个体对应的分量应替换为变异后的值
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
         # 保证每个个体都至少有一维分量由变异后的个体贡献
        mask[np.random.randint(0,7)]=True
        # 根据mask数组，对种群进行交叉操作，U为交叉后的种群
        self.U = np.where(mask, self.V, self.X)

    # Calculating the objective values of all individuals
    def x2y(self):
        self.Y = self.func(self.X)
        return self.Y

    # 此函数用来对种群进行选择操作
    def selection(self):
        X = self.X.copy()
        # 初始种群每个个体对应的适应值
        if  len(self.Y)!=0:
            X_Y = self.Y.copy()
        else:
            X_Y = self.x2y().copy()
        self.X = U = self.U
        # 交叉后种群每个个体对应的适应值
        U_Y = self.x2y()
        # 根据适应值大小进行选择，并将X变为单列
        self.X = np.where((X_Y < U_Y).reshape(-1, 1), X, U)
        self.x2y()
        """ X = self.X.copy()
        # 初始种群每个个体对应的适应值
        X_Y = self.x2y().copy()
        self.X = U = self.U
        # 交叉后种群每个个体对应的适应值
        U_Y = self.x2y()
        # 根据适应值大小进行选择，并将X变为单列
        self.X = np.where((X_Y < U_Y).reshape(-1, 1), X, U) """

    # 算法模型运行
    def run(self):
        start = time.perf_counter()
        # 初始化种群
        self.crtbp()
        # 进行迭代
        for i in range(self.max_iter):
            self.mutation()     # 变异
            self.crossover()    # 交叉
            self.selection()    # 选择
            # 记录每次迭代时适应值中最小值的下标
            generation_best_index = self.Y.argmin()
            # 记录适应值最小对应的个体
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            # 记录最小适应值
            self.generation_best_Y.append(self.Y[generation_best_index])
            #判断是否达到收敛
            if(self.convergeTime==500 and i>20):
                count=0
                for j in range(i-10,i):
                    if(abs(self.generation_best_Y[j]-self.generation_best_Y[j-1])<0.00001):
                        count+=1
                if(count==10):
                    self.convergeTime=i

            # 记录迭代中所有适应值
            self.all_history_Y.append(self.Y)
            print(self.Y[generation_best_index])

        del self.generation_best_X[0]
        del self.generation_best_Y[0]

        elapsed = (time.perf_counter() - start)
        self.totalTime=elapsed
        
        print("time cost:", elapsed)
        # 记录所有迭代中的最大适应值的下标
        global_best_index = np.array(self.generation_best_Y).argmin()
        print(global_best_index)
        # 记录最大适应值对应的个体值
        global_best_X = self.generation_best_X[global_best_index]
        print(global_best_X)
        # 记录最大适应值
        global_best_Y = self.generation_best_Y[global_best_index]
        print(global_best_Y)
        # 返回最大适应值个体及其适应值
        return global_best_X, global_best_Y