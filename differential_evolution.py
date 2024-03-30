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
        # generation_best_X/Y：历次迭代后种群中适应值最高的个体及其适应值
        self.generation_best_X = [-99]
        self.generation_best_Y = [-99]
        self.all_history_Y=[]
        # 下列参数与函数输入同义
        self.func= self.func_transformer(func)  # 此处对func函数进行变换
        self.F = F
        self.n_dim=n_dim
        self.size_pop=size_pop
        self.max_iter=max_iter
        self.prob_mut=prob_mut
        # X为初始种群，V和U分别为变异和交叉后的种群，Y为所有个体的适应值
        self.X,self.V, self.U= None, None,None
        self.Y=[]
        self.convergeTime=500
        self.totalTime=0
        # 生成初始个体标准类型与取值范围
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)

    # 此函数用来对函数进行改变
    # 输入为函数
    # 输出为改变后的函数
    # 具体变换为：func函数只能对单组植物参数进行相应的适应值计算，转换后的函数可对多组作物参数进行适应值计算。转换后的函数输入为多组参数组成的列表，输出为对应的适应值组成的列表
    def func_transformer(self,func):

        def func_transformed(X):
            return np.array([func(x) for x in X])

        return func_transformed

    # 此函数用来对种群进行初始化
    def crtbp(self):
        # 生成初始种群，X为种群个体的集合
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))

    # 此函数用来对种群进行变异操作
    def mutation(self):
        X = self.X
        # 挑选用于变异的个体
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        # V为变异后的种群
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])
        # 此处的操作在相关书籍中未发现描述！！！
        # 再次生成一个随机初始种群
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        # 如果变异后种群中部分个体的部分分量过界，则返回一个新的随机种群内的对应值
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)

    # 此函数用来对种群进行交叉操作
    def crossover(self):
        # mask为一个bool型数组，为true的值表示种群个体对应的分量应替换为变异后的值
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
         # 保证每个个体都至少有一维分量由变异后的个体贡献
        mask[np.random.randint(0,7)]=True
        # 根据mask数组，对种群进行交叉操作，U为交叉后的种群
        self.U = np.where(mask, self.V, self.X)

    # 此函数用来计算种群所有个体的适应值
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