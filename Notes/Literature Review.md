# Literature Review

### Multi-manipulator System

#### Literature Categorization

##### Distributed Manipulators on a Product Line:

1. [Agent-based planning and control of a multi-manipulator assembly system](https://ieeexplore.ieee.org/abstract/document/772528): 提出了一个agent-based机械臂协作框架；
2. [Multi-Robotic Arms Automated Production Line](https://ieeexplore.ieee.org/abstract/document/8384639): 提出了一个具体的多机器人协作系统；任务：Gluing and assembly line；

##### Cooperating Manipulators Performing the Same Task

1. [Optimization techniques applied to multiple manipulators for path planning and torque minimization](https://www.sciencedirect.com/science/article/abs/pii/S0952197602000672)





1. [J. . -C. Fraile, C. J. J. Paredis, Cheng-Hua Wang and P. K. Khosla, "Agent-based planning and control of a multi-manipulator assembly system," *Proceedings 1999 IEEE International Conference on Robotics and Automation (Cat. No.99CH36288C)*, 1999, pp. 1219-1225 vol.2, doi: 10.1109/ROBOT.1999.772528.](https://ieeexplore.ieee.org/abstract/document/772528)

#### **Main ideas:**

+ 提出了一个多机械臂系统（MMS）的分布式planning和control架构，这个框架是基于Multi-Agent Systems（MAS）框架设计的。
+ 主要面向flexible（生产任务可以较容易改变）的装配任务（Assembly task），该架构提出了一套各agent的控制及交互策略
+ 提出了一个基于artificial potential fields的分布式轨迹规划算法

这篇文章是1999年发表在ICRA上的，已经比较老了，学一学里面的问题描述和建模方式就好。

MAS架构被广泛用于Distributed Artificial Intelligence问题。所谓Multi-agent approach，就是指系统中的每个agent只有**limited local knowledge**，但却要合作来达成一些local和**global**的优化目标。

之前关于MMS的研究，可以按以下两类来划分：

+ 如何将MAS框架部署到MMS系统上：[2](https://ieeexplore.ieee.org/abstract/document/351029) 将待装配的零件的各部件看作agents；[3](https://ieeexplore.ieee.org/abstract/document/407653) 将每个机械臂看作agents；[4](https://ieeexplore.ieee.org/abstract/document/506532) 在抓取任务中将机械臂的各个部分看作agents；
+ agents之间的沟通方式：[5](https://ieeexplore.ieee.org/abstract/document/606871) 采用contract-net协议（CNP）；[6](https://www.sciencedirect.com/science/article/abs/pii/0736584595000089) 采用blackboard architecture；

该flexible装配MMS系统的定义如图：

![image-20220331214811626](https://s2.loli.net/2022/03/31/s7rmalbeiIMYUJd.png)

off-line阶段的输入是待装配零件的机械结构，输出是比较上层（preliminary）的装配操作（没有细化到机械臂控制命令），例如Pick/Place或Insert操作。

on-line阶段是优化问题主要关注的，包括任务分配（allocation）和执行（execution）。任务分配将上层操作细化分配到每个机械臂上。

#### 机械臂场景建模

![image-20220331215331807](https://s2.loli.net/2022/03/31/riNDctOV3RBPowx.png)

Part Feeder是给每个机械臂送待装配的零部件的；Rotational Table用于在机械臂之间传送正在装配的物件。

#### Agent建模

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

#### 避碰规划

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