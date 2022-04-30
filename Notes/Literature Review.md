# Literature Review

## Multi-manipulator System

### 任务定义

任务概述：

性质：总装配任务可以分解成每个机械臂承担的子任务，且这些子任务不存在相互依赖（task dependencies）。子任务可以表征成各机械臂共同工作空间中的不同任务点，且工作空间中存在静态障碍物。

![Task Defination](https://s2.loli.net/2022/04/05/2FXgqQWvE8n6TbI.png)

因此，机械臂 Scheduling问题可以分解成两个子问题：

+ 导航每个机械臂到指定的位置（Cartesian Coordinates）
+ 全局给每个机械臂分别规划它们需要达到的位置

**对于第一个子问题：**

将每个机械臂导航到指定位置，涉及到需要计算给定笛卡尔坐标的逆运动学解。对于较简单的机械臂结构（冗余度小-DOF<=2），可以直接求解给定坐标的关节角[[7]](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)。对于较复杂的机械臂结构，求解逆运动学问题的解比较复杂，大致有如下几种方法：

+ 基于强化学习（无法直接得出坐标到角度的映射）
+ 基于神经网络拟合IK函数[[8]](https://www.sciencedirect.com/science/article/abs/pii/S0952197699000500)（数据集不够？）

+ 基于启发式搜索算法例如GA，ASA，EM（速度快但每一对坐标点之间都需要重新搜索）

**对于第二个子问题：**

在第一个子问题已经计算出机械臂到各个任务点的近似解以后，假设空间中存在多个任务点需要到达，如果将这些任务点分配个多个机械臂来保证目标函数（minimum cycle time, minimum energy consumption, collision-free）达到near-optimal？

+ 分别算出各个机械臂到达每个任务点的时间，使用传统的offline scheduling算法即可（如果机械臂达到每个任务点都有多个solution，这样的scheduling算法时间复杂度很大
+ 使用启发式算法，定义好对每一种特定的任务实现方式的error function（fitness function），采用迭代的方式来求解near-optimal
+ 将每个任务点作为state，根据目标设计奖励函数，利用强化学习解这个序列规划问题（也可以将两个子问题放在一起，构造一个end-to-end的强化学习问题）

### Literature Categorization

+ **Distributed Manipulators on a Product Line:**

1. [Agent-based planning and control of a multi-manipulator assembly system](https://ieeexplore.ieee.org/abstract/document/772528): 提出了一个agent-based机械臂协作框架；
2. [Multi-Robotic Arms Automated Production Line](https://ieeexplore.ieee.org/abstract/document/8384639): 提出了一个具体的多机器人协作系统；任务：Gluing and assembly line；

+ **Cooperating Manipulators Performing the Same Task**

1. [Optimization techniques applied to multiple manipulators for path planning and torque minimization](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)

+ **Scheduling for Multi-robot Manipulator System**

1. [Real-time Scheduling of Distributed Multi-Robot Manipulator Systems](https://cdnsciencepub.com/doi/abs/10.1139/tcsme-2005-0012)
2. [Optimization of the time of task scheduling for dual manipulators using a modified electromagnetism-like algorithm and genetic algorithm](https://link.springer.com/article/10.1007/s13369-014-1250-0)
2. [Fast Scheduling of Robot Teams Performing Tasks With Temporospatial Constraints](https://ieeexplore.ieee.org/abstract/document/8279546)

+ **Flexible Job Scheduling Problem**

1. [Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S1568494620301484)

### Distributed Manipulators

#### Agent-based planning and control of a multi-manipulator assembly system

[J. . -C. Fraile, C. J. J. Paredis, Cheng-Hua Wang and P. K. Khosla, "Agent-based planning and control of a multi-manipulator assembly system," *Proceedings 1999 IEEE International Conference on Robotics and Automation (Cat. No.99CH36288C)*, 1999, pp. 1219-1225 vol.2, doi: 10.1109/ROBOT.1999.772528.](https://ieeexplore.ieee.org/abstract/document/772528)

##### **Main ideas:**

+ 提出了一个多机械臂系统（MMS）的分布式planning和control架构，这个框架是基于Multi-Agent Systems（MAS）框架设计的。
+ 主要面向flexible（生产任务可以较容易改变）的装配任务（Assembly task），该架构提出了一套各agent的控制及交互策略
+ 提出了一个基于artificial potential fields的分布式轨迹规划算法

这篇文章是1999年发表在ICRA上的，已经比较老了，学一学里面的问题描述和建模方式就好。

MAS架构被广泛用于Distributed Artificial Intelligence问题。所谓Multi-agent approach，就是指系统中的每个agent只有**limited local knowledge**，但却要合作来达成一些local和**global**的优化目标。

之前关于MMS的研究，可以按以下两类来划分：

+ 如何将MAS框架部署到MMS系统上：[[2]](https://ieeexplore.ieee.org/abstract/document/351029) 将待装配的零件的各部件看作agents；[[3]](https://ieeexplore.ieee.org/abstract/document/407653) 将每个机械臂看作agents；[[4]](https://ieeexplore.ieee.org/abstract/document/506532) 在抓取任务中将机械臂的各个部分看作agents；
+ agents之间的沟通方式：[[5]](https://ieeexplore.ieee.org/abstract/document/606871) 采用contract-net协议（CNP）；[[6]](https://www.sciencedirect.com/science/article/abs/pii/0736584595000089) 采用blackboard architecture；

该flexible装配MMS系统的定义如图：

![image-20220331214811626](https://s2.loli.net/2022/03/31/s7rmalbeiIMYUJd.png)

off-line阶段的输入是待装配零件的机械结构，输出是比较上层（preliminary）的装配操作（没有细化到机械臂控制命令），例如Pick/Place或Insert操作。

on-line阶段是优化问题主要关注的，包括任务分配（allocation）和执行（execution）。任务分配将上层操作细化分配到每个机械臂上。

##### 机械臂场景建模

![image-20220331215331807](https://s2.loli.net/2022/03/31/riNDctOV3RBPowx.png)

Part Feeder是给每个机械臂送待装配的零部件的；Rotational Table用于在机械臂之间传送正在装配的物件。

##### Agent建模

该论文的创新点在于，除了将各机械臂视为agent以外，将该场景中负责其他辅助任务的模块都抽象成了同级别的agent。例如：

- Scheduler Agent：规划off-line里的子装配任务并将它们分配给机械臂；
- Manipulator Agent：将Scheduler给的任务转化为机械臂可执行的命令，并执行控制；
- Auxiliary Component Agent：辅助部分也作为Agent，例如上面提到的part feeder和rotation table；
- World State Agent：存储系统运行过程中的静态和动态信息，便于其余Agent存取；
- Trajectory Planning Agent：从全局上规划一条保证机械臂不发生碰撞的路径；
- Communication Agent：节点沟通模块，分发和收集Agent发布的信息（采用blackboard architecture）；

![image-20220331220622273](https://s2.loli.net/2022/03/31/dL319kZHDwbmIJv.png)

每一个Agent都具有以下三个模块：

+ Communication模块：主要处理两种信息：state message（系统各状态参数）和command message（控制命令）。采用blackboard机制在节点之间沟通信息。
+ Knowledge模块：主要包含两类信息：local信息（agent的capabilities和它自身的状态参数）；global信息（通过communication模块获取到的其他agent的数据）
+ Control模块：控制模块决定agent的行为。每个agent都有自己的行为，例如manipulator agent通过通信模块从trajectory planning agent那里获取到了具体的控制执行命令。

##### 避碰规划

避碰规划由trajectory planning agent计算完成，采用Manipulator Incremental Motion (MIM) 方法（1999年的老方法...）。

![image-20220331221938618](https://s2.loli.net/2022/03/31/QqUXAELaSpWcHNm.png)

每个Manipulator agent先根据Scheduler agent分配的任务计算自己的incremental motion，Trajectory Planning agent取回这些数据计算分析后给每个机械臂的执行顺序分配priority。Priority一旦确定，各Manipulator agent就开始计算下一帧的incremental motion。

manipulator incremental motion是通过artificial potential field的方法计算的，该方法定义了两种potentials：

+ attractive potential：从C-space算出当前configuration距离目标的距离
+ repulsive potential：从W-space算出机械臂离障碍的最小距离

将该轨迹规划问题建模成一个启发式的搜索问题：

搜索空间的每个state由以下维度表示：

+ 当前configuration到目标configuration的距离
+ 系统中各物体（包括障碍）的位置

每个state之间的距离用上述定义的potential function（attractive function - repulsive function）来表示。

选择一种搜索算法后，就能找到机械臂的最佳incremental motion（这是由Manipulator agent计算的）。

### Cooperating Manipulators

#### Optimization techniques applied to multiple manipulators for path planning and torque minimization

[Garg, Devendra P., and Manish Kumar. "Optimization techniques applied to multiple manipulators for path planning and torque minimization." *Engineering applications of artificial intelligence* 15.3-4 (2002): 241-252.](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)

##### Main ideas

+ 提出了一种多机械臂协作任务结构，并针对这种结构进行数学建模，建立目标优化问题（最小化过程中的动力矩之和 -> 最小化能量消耗）
+ 使用遗传算法（GA）和模拟退火算法（ASA）来求解这个优化问题，发现两个算法虽然都能收敛到全局最优，但ASA算法的时间效率更高。

**论文结构**：

+ 多机械臂/机器人工业系统的部署是**有必要的**（multiple robots make manufacturing systems more flexible and these systems become capable of handling more complex operations）
+ 列举了2002年以前各种做multi manipulators的控制算法（主要是fuzzy control, adaptive control 和 NLP/QP optimization）
+ 探讨解决一个以最小化能量消耗为目标的路径规划问题的方法，主要有Enumerative schemes，Random search，梯度下降的方法，遗传算法（Genetic Algorithm）和退火算法（Simulated Annealing）等

##### Genetic Algorithm

[【算法】超详细的遗传算法(Genetic Algorithm)解析 - 简书 (jianshu.com)](https://www.jianshu.com/p/ae5157c26af9)

遗传算法（Genetic Algorithm, GA）是模拟达尔文生物进化论的自然选择和遗传学机理的生物进化过程的计算模型，是一种通过模拟自然进化过程搜索最优解的方法。其主要特点是**直接对结构对象进行操作，不存在求导和函数连续性的限定**；具有内在的隐并行性和更好的全局寻优能力；采用概率化的寻优方法，不需要确定的规则就能自动获取和指导优化的搜索空间，自适应地调整搜索方向。

遗传算法以一种群体中的**所有个体为对象**，并利用随机化技术指导对一个被编码的参数空间进行高效搜索。

**流程：**

![image-20220402151744706](https://s2.loli.net/2022/04/02/sNDKtWUfJBxXFR4.png)

例如是求解一个最短路径问题：

+ **Coding**：寻找一个表现型（路径）到基因型（编码后的数据）的映射关系，这个映射关系对算法的求解效率和最优解的收敛精度有很大的影响。这个映射关系应该保证两个string被选中crossover后生成的新string对应的表现型也在原解空间中。
  - 二进制编码（一维到一维的映射，将原维度按量级细分进一步后便于GA操作）
  - 浮点编码
  - 符号编码
+ **Initialization**：随机生成多个初始string（variation尽可能地大）
+ **Decoding**：将上一步得到的string解码回表现型（路径）
+ **Selection**：用定义好的与目标函数有关的适应性函数（fitness function）对string进行评估，适应度越高的越有更高的概率被选择出来。选择方式有如下几种：
  + 轮盘赌选择（Roulette Wheel Selection）
  + 随机竞争选择（Stochastic Tournament）
  + 最佳保留选择
  + 无放回随机选择
  + 确定式选择
  + 无回放余数随机选择
  + 均匀排序
  + 最佳保存策略
  + 随机联赛选择
  + 排挤选择
+ **Crossover**：被选择出来的spring将自己的染色体（spring）复制，放进一个mating pool。这些spring随机相互组合。每一对spring按照如下方式发生交叉重组：
  + 单点交叉（One-point Crossover）：指在个体编码串中只随机设置一个交叉点，然后再该点相互交换两个配对个体的部分染色体；
  + 两点交叉（Two-point Crossover）：在个体编码串中随机设置了两个交叉点，然后再进行部分基因交换；
  + 均匀交叉（也称一致交叉，Uniform Crossover）：两个配对个体的每个基因座上的基因都以相同的交叉概率进行交换，从而形成两个新个体；
  + 算术交叉（Arithmetic Crossover）：由两个个体的线性组合而产生出两个新的个体。该操作对象一般是由浮点数编码表示的个体；
+ **Mutation**：将新形成的spring上某些位上的编码发生随机改变（可以类比强化学习里的exploration，总是保留一定的概率在序列中的某些帧去尝试新的动作）。以下变异算子适用于二进制编码和浮点数编码的个体：
  + 基本位变异（Simple Mutation）：对个体编码串中以变异概率、随机指定的某一位或某几位仅因座上的值做变异运算；
  + 均匀变异（Uniform Mutation）：分别用符合某一范围内均匀分布的随机数，以某一较小的概率来替换个体编码串中各个基因座上的原有基因值。（特别适用于在算法的初级运行阶段）；
  + 边界变异（Boundary Mutation）：随机的取基因座上的两个对应边界基因值之一去替代原有基因值。特别适用于最优点位于或接近于可行解的边界时的一类问题；
  + 非均匀变异：对原有的基因值做一随机扰动，以扰动后的结果作为变异后的新基因值。对每个基因座都以相同的概率进行变异运算之后，相当于整个解向量在解空间中作了一次轻微的变动；
  + 高斯近似变异：进行变异操作时用符号均值为Ｐ的平均值，方差为P**2的正态分布的一个随机数来替换原有的基因值；

![image-20220402154929859](https://s2.loli.net/2022/04/02/iPvmLSEK85Xqtxd.png)

GA算法较其他求解优化算法的优点是：

+ GA算法是对数据点的编码进行操作，而不是数据点本身
+ GA算法每次迭代都获得一群数据点（BP算法每次迭代只有一个点）
+ GA算法不需要求导或者其他信息，仅需要目标函数本身
+ GA算法是基于概率的，不是确定性算法，因此对于非凸优化问题也有机会收敛到全局最优。（GAs use probabilistic transition rules, not deterministic rules.)

GA的缺点就是即使到了最优点附近也可能会反复横跳，不稳定（与编码方式的设计也有关）。

##### Approach

###### 任务定义

对于单臂机器人系统，给定初始和目标点end effector的位置；对于双臂协作机器人系统，给定其中一个机械臂的end effector的位置，另一个机械臂通过master-slave模式来跟随第一个机械臂end effector的路径。因此，无论是对于单臂还是双臂系统，都只由一个机械臂的两个关节角就能确定整个系统的状态（自由度均为2）。

![image-20220402185539822](https://s2.loli.net/2022/04/02/qbV2r8tz3YeS7uv.png)



![image-20220402185555045](https://s2.loli.net/2022/04/02/ZWOabfp2tj7IkD9.png)

###### 问题建模

为了便于使用GA算法对最优路径进行求解（尽可能降低描述问题所需的维度且保证crossover以后的string仍在原解集中），将关节角表示为时间的四次多项式：

![image-20220402192606954](https://s2.loli.net/2022/04/02/hgv65M2m9Do4RfK.png)

通过代入初始点和目标点的角度和角速度的边界条件，可以将后面的四个参数表示成参数a的一个函数。因此，仅需要将每个关节角的a参数（共两个参数）作为变量求解该优化问题即可。对于双机械臂系统，需要优化的变量还要多一个两机械臂之间的内力（也是通过类似的方式将各组各方向的内力关系表示成一个变量的函数）。

因此，对于单机械臂系统，需要优化的变量仅有两个；对于双机械臂系统，需要优化的变量有三个（变量数量越少，问题维度越低，使用GA和ASA算法的复杂度越低，效率越高）。

###### 优化目标

系统的最终优化目标是降低完成任务所需的总能量。而能量消耗是动力矩的正相关函数，因此只要保证在一次轨迹中所需要的力矩之和被最小化即可。

在这个问题中，优化目标和适应性函数（fitness function）相同

**单机械臂**

![image-20220402193959580](https://s2.loli.net/2022/04/02/kea3nULWK6Ot92R.png)

where τ1 and τ2 are the actuator torques applied at joints 1 and 2, respectively.

**双机械臂**

![image-20220402194448493](https://s2.loli.net/2022/04/02/65xeQhnuvS3KNtg.png)

actuator torque和关节角的关系是：

![image-20220402194215169](https://s2.loli.net/2022/04/02/2KnAqlIeMmofF6z.png)

因此，只需要将轨迹函数（角度与时间的关系函数）在时间上进行离散化，再代入上述关系式，即可将各待优化参数表示成fitness function的函数。此时再使用GA和ASA等方法分别去优化求解即可。

###### 仿真结果

初始位置和目标位置：

![image-20220402194609701](https://s2.loli.net/2022/04/02/CZtTEzlujkU1rf8.png)

The arrival time has been specified to be equal to 2 s.

**单机械臂：**

![image-20220402194530865](https://s2.loli.net/2022/04/02/LEVgToG3asIKkQl.png)

单机械臂的两个link在空间的移动示意图（从两个角度都为0 rad，到都为1 rad）

![image-20220402194838771](https://s2.loli.net/2022/04/02/D2bXZcGpNEg9OdK.png)

每个generation（每次迭代的多个string为一个generation）的平均PI

![image-20220402195052056](https://s2.loli.net/2022/04/02/16YZHV7IPpAEODT.png)

单位时间（不一定按照generation来划分）存在的最低PI（当前获得的最优解）

**双机械臂：**

![image-20220402195222368](https://s2.loli.net/2022/04/02/VkQf2dMHclKtshj.png)

![image-20220402195257274](https://s2.loli.net/2022/04/02/WkY3DALChUO9Qqr.png)

###### 结论

1. Both the algorithms (GA and ASA) reached to same solution which confirms that the solution found was a **global minimum**.
2. GA are mathematically less complex, and relatively simple and easy to code. The above application, however **shows faster convergence rate of ASA as compared to GA**.

这篇文章在描述机械臂轨迹时采用多项式函数拟合来降低原本有无穷多轨迹的搜索空间（可以类比拟合KDE时使用的最大似然法）。该优化问题的解空间维度大大降低后，才使得GA和ASA算法得以发挥作用。这样采用简单模型来简化复杂问题的描述方式值得学习。

### Scheduling Problem

#### Real-time Scheduling of Distributed Multi-Robot Manipulator Systems

[P. Yuan, M. Moallem, and R.V. Patel. REAL-TIME SCHEDULING OF DISTRIBUTED MULTI-ROBOT MANIPULATOR SYSTEMS. *Transactions of the Canadian Society for Mechanical Engineering*. **29**(2): 179-194. https://doi.org/10.1139/tcsme-2005-0012](https://cdnsciencepub.com/doi/abs/10.1139/tcsme-2005-0012)

##### Main ideas:

+ Multi-robot这样的实时系统需要real-time scheduler, which depends on the **dispatching mechanism**:
  + time-driven:
    + Branch and bound search
  + priority-driven:
    + First-In-First-Out (FIFO)
    + Round-Robin (RR)
    + Earliest-Deadline-First (EDF)
    + Minimum-Laxity-First (MLF)
    + Least-Slack-Time-First (LST)
+ 上述算法都没有考虑在multi-robot装配系统中至关重要的task dependencies（即任务之间的依赖关系）
+ 该论文提出了一个考虑了任务间依赖的task-oriented的on-line scheduling算法和一个可以避免deadlock的offline scheduling算法。

#### Optimization of the Time of Task Scheduling for Dual Manipulators using a Modified Electromagnetism-Like Algorithm and Genetic Algorithm

[Abed, I. A., Koh, S. P., Sahari, K. S. M., Jagadeesh, P., & Tiong, S. K. (2014). Optimization of the time of task scheduling for dual manipulators using a modified electromagnetism-like algorithm and genetic algorithm. *Arabian Journal for Science and Engineering*, *39*(8), 6269-6285.](https://link.springer.com/article/10.1007/s13369-014-1250-0)

#### Main ideas

机械臂 Scheduling问题可以分解成两个子问题：

+ 导航每个机械臂到指定的位置（Cartesian Coordinates）
+ 全局给每个机械臂分别规划它们需要达到的位置

**对于第一个子问题：**

将每个机械臂导航到指定位置，涉及到需要计算给定笛卡尔坐标的逆运动学解。对于较简单的机械臂结构（冗余度小-DOF<=2），可以直接求解给定坐标的关节角[[7]](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)。对于较复杂的机械臂结构，求解逆运动学问题的解比较复杂，大致有如下几种方法：

+ 基于强化学习（无法直接得出坐标到角度的映射）
+ 基于神经网络拟合IK函数[[8]](https://www.sciencedirect.com/science/article/abs/pii/S0952197699000500)（数据集不够？）

+ 基于启发式搜索算法例如GA，ASA，EM（速度快但每一对坐标点之间都需要重新搜索）

**对于第二个子问题：**

在第一个子问题已经计算出机械臂到各个任务点的近似解以后，假设空间中存在多个任务点需要到达，如果将这些任务点分配个多个机械臂来保证目标函数（minimum cycle time, minimum energy consumption, collision-free）达到near-optimal？

+ 分别算出各个机械臂到达每个任务点的时间，使用传统的offline scheduling算法即可（如果机械臂达到每个任务点都有多个solution，这样的scheduling算法时间复杂度很大
+ 使用启发式算法，定义好对每一种特定的任务实现方式的error function（fitness function），采用迭代的方式来求解near-optimal
+ 将每个任务点作为state，根据目标设计奖励函数，利用强化学习解这个序列规划问题（也可以将两个子问题放在一起，构造一个end-to-end的强化学习问题）

#### IK Solvation

对于第一个子问题，这篇paper使用了他们升级后的**EM**算法，the so-called Modified EM Algorithm with Two-Direction Local Search (MEMTDLS).

EM算法：[智能优化算法：类电磁机制算法 - 附代码_智能算法研学社（Jack旭）的博客-CSDN博客](https://blog.csdn.net/u011835903/article/details/120902972?spm=1001.2014.3001.5506)

给定任务点（三维坐标）为输入，他们使用EM算法求解一组较好的近似IK关节角解。相关目标函数定义如下：

![image-20220405153824523](https://s2.loli.net/2022/04/05/PlRaYmZOX3bcL4k.png)

![image-20220405153839450](https://s2.loli.net/2022/04/05/W4SIBveLYMJTnjz.png)

两个任务点之间的trajectory直接使用3次多项式拟合：

![image-20220405154001664](https://s2.loli.net/2022/04/05/mT1bQLkj8xHMarV.png)

和上一篇论文不同的是，这里的四个常数不是要优化的变量，直接求解即可。上一篇论文中，准确的关节角已经给出，需要plan的就是两个角度之间功耗最低的trajectory（Actually, 上一篇论文的所有研究都focus在了两个角度之间的最优trajectory上，而这篇论文已经预设了这样的trajectory，研究目标并不相同）。

![image-20220405161823607](https://s2.loli.net/2022/04/05/Ne9ZPYwCf5hKuWG.png)

原本EM算法的local search是对于每一个迭代出的解都在邻域（length是固定的）随机搜索一个更好的解：

![image-20220405154541715](https://s2.loli.net/2022/04/05/KmurpXjOg476NJ3.png)

My comments: This pseudo code is really confusing. The study of Science is decayed exactly by papers like these. 
I think "Counter" is identical to "counter". The maximum iterations of local search shall not surpass Lsiter, so for those randomly generated y, if their corresponding f value do not fare better than the original θi, counter is incremented by 1(until it reaches Lsiter). Otherwise, counter jumps to Lsiter - 1, and then incremented by 1, the loop is finished immediately.

m是数据点的个数；n是维度；Lsiter是对于每个维度在邻域内随机搜索的最高迭代次数（一旦找到更优点就直接停止对此数据点此维度的迭代）因此，时间复杂度是 m * n * Lsiter。

改进后的EM算法仅在上一次迭代得到的best point的邻域内搜索，且邻域的length随迭代此输的增多而递减。

![image-20220405154947816](https://s2.loli.net/2022/04/05/SKaw1JWUOckpnER.png)

因为他们发现，length越小，收敛精度越高（怎么会这样？）

![image-20220405155111537](https://s2.loli.net/2022/04/05/8Wlahw2UJrndMkq.png)

并且，在改进的算法中，对于每一个维度上的数值在邻域内改变的方向（增或减）并不是随机的，而是将两个方向改变后的新数据点对应的值都保存下来，再最后进行比较。

![image-20220405155336493](https://s2.loli.net/2022/04/05/ORLDC8yuJIMG3AV.png)

从上述伪代码中易得，时间复杂度是m * k (* 2)。因为对于每个维度k，仅在两个方向上各跨出λ*β，产生一对新的数据点。这个改进后的算法仅在最优点附近搜索，确实可能保证更快收敛到更优的值。

![image-20220405161455334](https://s2.loli.net/2022/04/05/e8194sykrKI3L2g.png)

![image-20220405161514712](https://s2.loli.net/2022/04/05/SLrOkBYe2CsptEX.png)

值得注意的是，使用EM算法来求解给定坐标点的关节角，虽然一直强调best point，但最后却保留了multiple solutions。（由于高DOF机械臂的冗余，达到相似的coordinate，joint angles可能vary greatly，导致到达这些角度所需要的时间也vary greatly。所以最精确的角并不代表最temporal-efficient，因此需要把multiple joints都保留下来供后面的GA Scheduler来选择）

#### GA Scheduler

对于第二个子问题，该论文采用启发式算法GA来给每个机械臂分配任务点。

给定mα个任务点以及他们对应的solution（EM求出的multiple solutions中的某一个），travel time:

![image-20220405160241949](https://s2.loli.net/2022/04/05/r1EKhgjCuWmMq8l.png)

从mα回到初始点（优化目标是cycle time）所需的travel time:

![image-20220405160356359](https://s2.loli.net/2022/04/05/bAW5q4VoX7rhcpk.png)

对于某一个机械臂，cycle time:

![image-20220405160452824](https://s2.loli.net/2022/04/05/Qb1fRp93tWM2cVB.png)

因此，对于一个双机械臂场景，需要优化的方程是：

![image-20220405160529030](https://s2.loli.net/2022/04/05/b4ys9I6mrBqQWVL.png)

考虑到避碰问题，将机械臂和环境中的静态障碍物采用tangent circle方法来建模：

![image-20220405160701935](https://s2.loli.net/2022/04/05/O3jB271szurUpGI.png)

因此，若在某一次特定规划中发生了碰撞，则将原优化目标乘一个很大的数来迫使最优规划方案中不发生碰撞：

![image-20220405160849719](https://s2.loli.net/2022/04/05/3wSh1Cd9Yacs6BG.png)

使用GA算法，以TF为fitness function来求解两个机械臂分配得到的任务点、这些任务点的到达顺序以及各任务点采用哪种solution。

![image-20220405161045072](https://s2.loli.net/2022/04/05/RSN8MVy3dthe1nQ.png)

在一个chromesome中，前8位代表任务点的到达顺序；中间8位表示前8个任务点具体采用的solution；最后一位代表总任务的divider：例如若divider为3，则前三位对应的任务分配个1号机械臂，后五位对应的任务分配给2号机械臂。

对于这三个组成部分，这篇论文都定义了恰当的crossover和mutation方式，来适应这个特定的任务。对于这样的离散规划问题，使用GA是有一定优势的。

#### Fast Scheduling of Robot Teams Performing Tasks With Temporospatial Constraints

[Gombolay, M. C., Wilcox, R. J., & Shah, J. A. (2018). Fast scheduling of robot teams performing tasks with temporospatial constraints. *IEEE Transactions on Robotics*, *34*(1), 220-239.](https://ieeexplore.ieee.org/abstract/document/8279546)

##### Main Idea

1. 提出了"Tercio"算法来解决有时间（temporal）和空间（spatial）约束的机械臂任务分配（task allocation）问题。该算法的输出是每个agent分配得到的task和这些task的sequence。
2. 该算法由一个multi-agent task sequencer（启发自计算机处理器的scheduling techniques）和一个混合整数线性规划器（mixed-integer linear program solver）组成。



##### 混合整数线性规划（MILP）方法

任务规划问题可以化归为一个MILP问题。解决MILP问题常用的方法有Benders Decomposition。

###### Benders Decomposition

Benders分解由Jacques F. Benders在1962年提出[[9]](https://blog.csdn.net/qx3501332/article/details/104978928/#fn1)。 它是一种把线性规划问题分解为小规模子问题的技巧。通过迭代求解主问题和子问题，从而逼近原问题的最优解。

[线性规划技巧: Benders Decomposition_胡拉哥的博客-CSDN博客](https://blog.csdn.net/qx3501332/article/details/104978928/)

[Benders分解(Benders Decomposition)算法：算法原理+具体算例_出淤泥的博客-CSDN博客_benders分解算法](https://blog.csdn.net/weixin_42991799/article/details/120247976)

[Benders分解算法详解_大弱智鱼的博客-CSDN博客_benders分解算法](https://blog.csdn.net/qq_42756072/article/details/107728357)

之所以要使用Benders分解，就是因为原问题中既有较复杂的约束（整数约束、非线性约束）又有线性约束，难以直接求解。因此motivation就是：能否将变量解耦，把原问题分成主问题（仅包含约束较复杂约束的变量）和子问题（仅包含线性约束）。子问题是一个标准的线性规划问题。Benders通过迭代不断引入约束提高原问题的下界的方式最后逼近最优解。

原问题：![image-20220413155421636](https://s2.loli.net/2022/04/13/cTqbYBNw26avL1I.png)

主问题（master problem）：![image-20220413155445600](https://s2.loli.net/2022/04/13/sKhZHB7UpaDwXxl.png)

子问题（subproblem）：![image-20220413155506513](https://s2.loli.net/2022/04/13/71RIjoeqOtBEYNy.png)

子问题的对偶问题：![image-20220413155631836](https://s2.loli.net/2022/04/13/H1AuRkVgfSLJmNx.png)

之所以要求子问题的对偶问题，是因为子问题中可行域与主问题中fix的y有关，而其对偶问题的可行域并不依赖于y，y只影响目标函数值。因此，若对偶问题有界，则其极值一定存在于![image-20220413155905272](https://s2.loli.net/2022/04/13/omR3lu8dst4Yr1p.png)所确定的多面体（非有限多面体，即非封闭的）的顶点处（或者Benders optimality cuts里描述的极点)。因为该多面体的顶点（极点)是有限N个的，因此可以分别将每个顶点求出代入到原问题中构造出N个约束条件。即写成如下形式：

![image-20220413160208673](https://s2.loli.net/2022/04/13/EmzFHt1q6DJhW2R.png)

q是将每个顶点分别代入对偶问题得到的最大值，即子问题在主问题fix y的情况下的解。如果不求对偶问题，则子问题的极值难以表示。对偶问题将子问题的约束变成了有限个，使得我们可以不用一开始就求得很复杂的主问题，而可以慢慢增加约束数量（通过引入顶点或极线），至少能保证在较少计算量的情况下就得到near-optimal的解。

上式中，![image-20220413160645400](https://s2.loli.net/2022/04/13/k64VH3XOmqi2Cag.png)代表极线（extreme rays)。因为主问题的约束数量是慢慢增加的，因此每次求得的(y, q)不一定是满足所有约束条件的。这样求主问题得到的y代入子问题，子问题可行域可能为空。子问题的可行域为空，则对偶问题无界；子问题无界，则对偶问题可行域为空。因此，当且仅当，对偶问题有界时：

![image-20220413160618402](https://s2.loli.net/2022/04/13/vBXbqgT5Lz3HMek.png)

主问题求得的y才能使得子问题的可行域有解（这样的x,y构成一组可行解）。当y0代入子问题后，对偶问题无界时（说明子问题可行域为空），我们应该对主问题添加一个新的约束，来保证再次求解主问题的时候，不会再求得y0了。这样的约束就由上述的极线表达式确定。

[在最优化里面，如何理解多面体的极点与极方向？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/376887239/answer/1077977910)

上述关于极点和极线的描述看似抽象，我认为可以这么理解：

在一个二维平面上，假设某可行域为[0,+inf] 则极点为(0,0)，极方向为(1,0）或x>0。假设需要优化的函数为 max(y) = -x+2 。则显然，此时满足式：

![image-20220413160618402](https://s2.loli.net/2022/04/13/eUlZzXCbOvxidpF.png)

因为![image-20220413160645400](https://s2.loli.net/2022/04/13/67f21gN9yZvjKS4.png)>0，(b-By) = -1。最大值在极值处取得。但如果主问题fix的y使得子问题的对偶问题的待优化函数变成了：max(y) = x+2 。显然此时该问题无界，因为此时：![image-20220413164012454](https://s2.loli.net/2022/04/13/qOmBGzYJapthe65.png)。

在求解该对偶问题时，如果出现![image-20220413164012454](https://s2.loli.net/2022/04/13/3lYDoEL4IJcrSeA.png)的情况，可以求出该情况对应的极线，然后将该约束加入主问题中。

通过往主问题中不断增加约束，可以保证原问题的下界不断增加。对于仅含有部分约束的主问题中，若得到的y *（对应的q *仅为部分顶点对应的最大值）代入子问题中（代入所有顶点去求最大值），得到的q比q *更大，说明对于y * 在顶点集中还有顶点需要加入主问题中去作为约束条件。如果得到的y *代入子问题后，得到的q和q *相等，说明主问题中的约束已经完全包含了求得该q的所有顶点，此时得到最优解。也即是说，若主问题连续两次求得的y相同时（此时计算得到的q和q *一定相等），得到最优解。

因为在迭代过程中，下界是单调递增的。每次代入子问题后可以求得一次原问题的值，在这些值中取得最小值即是原问题的上界。反复求解主问题和子问题直到上下界相等（或非常接近）时，取得使得原问题取上界的解即为optimal solution（上下界相等），near-optimal solution（非常接近）。

**Logic-Based Benders Decomposition**



**Constraint Programming**

[约束满足问题（CSP）_宇内虹游的博客-CSDN博客_约束满足问题](https://blog.csdn.net/weixin_39278265/article/details/80932277)

**使用要素化来描述状态**：一组变量，每个变量有自己的值。当每个变量都有自己的赋值同时满足所有关于变量的约束时，问题就得到了解决。这类问题就叫做约束满足问题（CSP），全称Constraint Satisfaction Problem。

CSP利用了状态结构的优势，使用的是**通用策略**而不是**问题专业启发式**来求解复杂问题。

**主要思想**：通过识别违反约束的变量/值的组合迅速消除大规模的搜索空间(约束传播算法）。

**约束传播：**

+ 节点相容：单个变量（对应一个节点）值域中的所有取值满足它的一元约束，就是节点相容的；
+ 弧相容：如果CSP中某变量值域中所有取值满足该变量所有二元约束，则此变量弧相容；如果每个变量相对其他变量都是弧相容的，则称该网络是弧相容的；
+ 路径相容：观察变量得到隐式约束，并以此来加强二元约束；

![这里写图片描述](https://s2.loli.net/2022/04/28/oW38bdg2SQkLV9P.png)

+ k-相容：如果对于任何k-1个变量的相容赋值，第k个变量总能被赋予一个和前k-1个变量相容的值，那么这个CSP就是k相容的；（推导：2-相容==该网络是弧相容的）；

在澳大利亚地图着色问题中，显然该网络都是弧相容的，因此意义不大，需要更强的相容概念：路径相容。





#### Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning

动态任务Balancing and Scheduling主要有下面三种应对策略：

+ **Mathematical Programming**: 用MILP+CP等数学规划的方式强行解出最优解，这种方案往往能保证最终solution的optimality，但是只能应对小规模的问题（NP hard problem）；
+ **Dispatching Rules**: 根据人为先验知识提出一些dispatching rules，这些rules能够立即得出dynamic场景下的每个job的assignment和scheduling方案，但这个方法是myoptic的，甚至不能保证局部最优；
+ **Meta-heuristics**: Meta-heuristics将动态规划问题分解成一系列静态的子问题，然后分别用一些启发式算法 (e.g. GA, SA, PSO) 来求解；这样的方法得到的解虽然比dispatching rules更优（考虑了更长的time window)，但更time-consuming，因此不适合real-time scheduling；
+ **Reinforcement Learning**: DFJSP问题是一个典型的Markov Decision Process，因此可以用RL来解这个MDP过程；RL能够快速对dynamic events进行反应，而且由于训练过程经验的积累，RL能够比dispatching rules更高效地选择利于全局最优的dispatching方案；RL和其他几种方法一样，无法保证得到最优解，但由于它能对环境和任务的uncertainty更高效地反应，使得它成为目前最流行的DFJSP方法；



此处需要注意两种