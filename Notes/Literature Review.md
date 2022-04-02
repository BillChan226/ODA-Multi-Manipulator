# Literature Review

## Multi-manipulator System

### Literature Categorization

+ **Distributed Manipulators on a Product Line:**

1. [Agent-based planning and control of a multi-manipulator assembly system](https://ieeexplore.ieee.org/abstract/document/772528): 提出了一个agent-based机械臂协作框架；
2. [Multi-Robotic Arms Automated Production Line](https://ieeexplore.ieee.org/abstract/document/8384639): 提出了一个具体的多机器人协作系统；任务：Gluing and assembly line；

+ **Cooperating Manipulators Performing the Same Task**

1. [Optimization techniques applied to multiple manipulators for path planning and torque minimization](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)



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

+ **Coding**：寻找一个表现型（路径）到基因型（编码后的数据）的映射关系，这个映射关系对算法的求解效率和最优解的收敛精度有很大的影响。这个映射关系应该保证两个spring被选中crossover后生成的新spring对应的表现型也在原解空间中。
  - 二进制编码（一维到一维的映射，将原维度按量级细分进一步后便于GA操作）
  - 浮点编码
  - 符号编码
+ **Initialization**：随机生成多个初始spring（variation尽可能地大）
+ **Decoding**：将上一步得到的spring解码回表现型（路径）
+ **Selection**：用定义好的与目标函数有关的适应性函数（fitness function）对spring进行评估，适应度越高的越有更高的概率被选择出来。选择方式有如下几种：
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

GA算法较其他求解优化算法的优点是：

+ GA算法是对数据点的编码进行操作，而不是数据点本身
+ GA算法每次迭代都获得一群数据点（BP算法每次迭代只有一个点）
+ GA算法不需要求导或者其他信息，仅需要目标函数本身
+ GA算法是基于概率的，不是确定性算法，因此对于非凸优化问题也有机会收敛到全局最优。（GAs use probabilistic transition rules, not deterministic rules.)

GA的缺点就是即使到了最优点附近也可能会反复横跳，不稳定（与编码方式的设计也有关）。