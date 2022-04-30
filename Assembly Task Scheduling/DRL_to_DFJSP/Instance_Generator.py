import random
import numpy as np

Total_Machine=[10,20,30,40,50]  #全部机器
Initial_Job_num=20              #初始工件个数
Job_insert=[50,100,200]         #工件新到达个数
DDT=[0.5,1.0,1.5]               #工件紧急程度
E_ave=[50,100,200]              #指数分布

def Instance_Generator(M_num,E_ave,New_insert,DDT):
    '''
    :param M_num: Machine Number
    :param E_ave: exponetional distribution
    :param New_insert: New Job insert
    :param DDT:DDT
    :return: Processing time,A:New Job arrive time,
                                D:Deliver time,
                                M_num: Machine Number,
                                Op_num: Operation Number,
                                J_num:Job NUMBER
    '''
    Initial_Job_num=5
    Op_num=[random.randint(1,5) for i in range(New_insert+Initial_Job_num)] # Number of operations for each job
    print('Op_num', Op_num)
    Processing_time=[]
    for i in range(Initial_Job_num+New_insert):
        Job_i=[]
        for j in range(Op_num[i]):
            k=random.randint(1,M_num-2) # each operation can be operated on at least 3 machine, at most 10 machines
            T=list(range(M_num))
            random.shuffle(T)
            T=T[0:k+1]
            O_i = list(np.ones(M_num) * (-1))
            for M_i in range(len(O_i)):
                if M_i in T:
                    O_i[M_i]=random.randint(1,50)
            Job_i.append(O_i)
        Processing_time.append(Job_i)
    A1=[0 for i in range(Initial_Job_num)] # for jobs initially presented, their arrival time should naturally be 0
    A=np.random.exponential(E_ave, size=New_insert) # E_ave is the scale(mean) parameter
    A=[int(A[i]) for i in range(len(A))] # New Insert Job arrive time
    A1.extend(A)
    T_ijave = []
    for i in range(Initial_Job_num+New_insert):
        Tad = []
        for j in range(Op_num[i]):
            T_ijk = [k for k in Processing_time[i][j] if k != -1]
            Tad.append(sum(T_ijk) / len(T_ijk))
        T_ijave.append(sum(Tad))
    D1=[int(T_ijave[i]*DDT) for i in range(Initial_Job_num)]
    D=[int(A1[i]+T_ijave[i]*DDT) for i in range(Initial_Job_num,Initial_Job_num+New_insert)]
    D1.extend(D)
    O_num=sum(Op_num)
    J=dict(enumerate(Op_num))
    J_num=Initial_Job_num+New_insert

    return Processing_time,A1,D1,M_num,Op_num,J,O_num,J_num

#Processing_time,A,D,M_num,Op_num,J,O_num,J_num=Instance_Generator(10,50,10,0.5)
#print(Processing_time,A,D,M_num,Op_num,J,O_num,J_num)

#print(J)
#print(np.shape(D))

# J_num = 3
# O_num = 12
# Op_num = [4, 4, 4]
# M_num = 3
# A = [0, 0, 0]
# D = [100, 100, 100]
# J = {0: 4, 1: 4, 2:4}
# Processing_time=[
#     [
#         [4, 4, 4], [2, 2, -1], [2, 2, 2], [2, -1, 2]
#     ],
#     [
#         [4, 4, 4], [2, 2, -1], [2, -1, 2], [4, 4, 4]
#     ],
#     [
#         [2, 2, -1], [2, 2, 2], [2, -1, 2], [4, 4, 4]
#     ]
# ]

J_num = 5
O_num = 38
Op_num = [6, 8, 8, 8, 8]
M_num = 3
A = [0, 0, 0, 0, 0]
D = [100, 100, 100, 100, 100]
J = {0: 6, 1: 8, 2:8, 3:8, 4:8}
Processing_time=[
    [
        [4, -1, 4], [2, -1, 2], [2, -1, 2], [2, 2, -1], [2, 2, -1], [-1, 5, 5]
    ],
    [
        [4, -1, 4], [2, -1, 2], [2, 2, -1], [4, 4, -1], [2, 2, -1], [-1, 3, 3], [-1, 2, 2], [-1, 4, 4]
    ],
    [
        [2, -1, 2], [2, -1, 2], [2, 2, -1], [4, 4, -1], [-1, 3, 3], [-1, 5, 5], [-1, 2, 2], [-1, 4, 4]
    ],
    [
        [4, -1, 4], [2, -1, 2], [4, 4, -1], [2, 2, -1], [-1, 3, 3], [-1, 5, 5], [-1, 2, 2], [-1, 4, 4]
    ],
    [
        [4, -1, 4], [2, -1, 2], [4, 4, -1], [2, 2, -1], [-1, 3, 3], [-1, 5, 5], [-1, 2, 2], [-1, 4, 4]
    ]
]
