# 朴素贝叶斯法

## 1. 预备知识
1. 先验概率: 用$P(B_{i})$表示没有训练数据前假设假设h拥有的初始概率，称为先验概率。先验概率反映了关于$B_{i}$是一正确假设的机会的背景知识
2. 后验概率:$P(B_{i}|A)$为后验概率, 给定$A$时$B_{i}$成立的概率, 称为$B_{i}$的后验概率。拿在朴素贝叶斯分类时, 对给定的输入$x$, 通过学习到的模型计算后验概率分布$P(Y=c_{k}|X=x)$, 将后验概率最大的类作为$x$的类的输出。
3. 极大后验概率: 最佳假设$P(B_{k}|x)=argmin_{k} \frac {P(x|B_{i})P(B_{i})} { P(x) }$
4. 极大似然估计:假设$H$中的每个假设没有相同的先验概率, 可以简化为:$P(B_{k}|x)=argmin_{k}P(x|B_{i})$

## 2. 概述
1. 朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法
2. 对于给定的数据集, 首先基于特征条件独立假设学习输入$/$输出的联合概率分布
3. 接着根据此模型也就是输入$/$输出的联合概率分布,然后对于给定的输入`x`, 利用贝叶斯定理求出后验概率中最大的输出`y`  
4. 找到一个已知分类的待分类项集合，这个集合称为训练样本
## 3. 贝叶斯定理
1. 朴素贝叶斯法对条件概率分布作了条件独立性的假设,具体的条件独立性假设是:$P(X=x|Y=c_{k})=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},...,X^{(K)}=x^{(K)}|Y=c_{k}),k=1,2,....,K$
2. 贝叶斯定理计算后验概率:$\mathrm{P}\left(\mathrm{Y}=c_{k} | \mathrm{X}=\mathrm{x}\right)=\frac{P\left(X=x | Y=c_{k}\right) P\left(Y=c_{k}\right)}{\sum_{k} P\left(X=x | Y=c_{k}\right) P\left(Y=c_{k}\right)}$, 其中$P=(Y=c_{k})$是先验概率，根据数据集可以求得:$P(Y=c_{k}|X=x)$是后验概率，也是条件概率,在已知输入的情况下求输出$Y=c_{k}$的概率;$P(X=x|Y=c_{k})$是条件概率,即在$Y=c_{k}$发生情况下, $X=x$的概率。
3. 由于以上分式, 分母项都相同, 所以求分子项最大即可,即$\mathrm{y}=\operatorname{argmax} \mathrm{P}\left(\mathrm{Y}=c_{k}\right) \mathrm{P}\left(\mathrm{X}=\mathrm{x} | \mathrm{Y}=c_{k}\right)$,其中, $X$是特征向量, 其有多个维度,即 $X={x_{1},x_{2},x_{3}...,x_{n}}$。假设各个特征取值是相互独立的，可得下式:$\mathrm{P}\left(\mathrm{X}=\mathrm{x} | \mathrm{Y}=c_{k}\right)=\prod_{j}^{n} P\left(X^{j}=x^{j} | Y=c_{k}\right)$
## 4. 朴素贝叶斯算法
1. 设$x={a_{1},a_{2},...,a_{m}}$为一个待分类项,其中$a_{i}$为$x$的一个特征属性
2. 有类别集合$C=\left \{  y_{1},y_{2},...,y_{n} \right \}$
3. 计算后验概率$P(y_{1}|x),P(y_{2}|x),...,P(y_{n}|x)$
4. 若$P(y_{k}|x)=max_{k}\left \{ P(y_{1}|x),P(y_{2}|x),...,P(y_{n}|x) \right \}$,则将$x$分类到第$k$类

## 5. 朴素贝叶斯的基本思路
- 设输入空间$x\in \chi \subseteq R^{n}$, 输出空间$y\in \nu ={c_{1},c_{2},...,c_{k}}$为$\chi$上的随机向量, $Y$是定义在$V$上的随机向量,$P(X,Y)$是$X$和$Y$的联合分布分布;数据集$T={(x_{1},y_{1}),(x_{2},y_{2}),...,(x_{N},y_{N})}$T=(x1,y1),(x2,y2),...,(xN,yN)由$P(X,Y)$独立且同分布产生;$P(X,Y)$由学习产生的先验概率分布及条件概率分布求解。 
  - **step1:**
    1. $P(Y=c_{k}),k=1,2,...,K  $为先验概率分布
    2. $P(X=x|Y=c_{k})=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},...,X^{(K)}=x^{(K)}|Y=c_{k}),k=1,2,....,K$为条件概率分布
    3. 由于条件概率分布$P(X=x|Y=c_{k})$有指数级数量的参数，其估计实际是不可行的，故朴素贝叶斯对条件概率分布做了条件独立性假设:$P(X=x|Y=c_{k})=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},...,X^{(K)}=x^{(K)}|Y=c_{k})=\prod_{i=1}^{n} P(X^{(i)}=x^{(i)}|Y=c_{k})$

  - **step2:**
  
     1. 朴素贝叶斯分类时，对给定的输入x，通过学习得到的模型计算后验概率分布$P(Y=c_{k}|X=x)$，将后验概率最大类作为x的类输出:$$P(Y=c_{k}|X=x)=\frac {P(Y=c_{k})P(X=x|Y=c_{k})} {\sum_{k=1}^{K}P(Y=c_{k})P(X=x|Y=c_{k})}$$
     2. 将**step1**中的3式代入**step2**中的1式中, 可得:$$P(Y=c_{k}|X=x)=\frac {P(Y=c_{k})P(X^{(j)}=x^{(j)}|Y=c_{k})} {\sum_{k=1}^{K}P(Y=c_{k})P(X^{(j)}=x^{(j)}|Y=c_{k})}$$
     3. 将上式转化为寻找最大后验概率:$y=f(x)=argmax_{c_{k}}P(Y=c_{k}|X=x)=\frac {P(Y=c_{k})P(X^{(j)}=x^{(j)}|Y=c_{k})} {P(x)}$, 将$P(x)$固定,即可得$y=argmax_{c_{k}}P(Y=c_{k})\prod_{i=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_{k})$这就是求解的最大后验概率
## 6. 后验概率分布最大化的含义
- 损失函数度量模型一次预测的好坏, 风险函数度量平均意义下模型预测的好坏
- 假设选择$0~1$损失函数:$$L(Y, f(X))=\left\{\begin{array}{ll}{1,} & {Y \neq f(X)} \\ {0,} & {Y=f(X)}\end{array}\right.$$这时, 期望风险函数为$$\begin{aligned} R_{e x p}(f) &=E[L(Y, f(X))] =E_{X} \sum_{k=1}^{K} L\left(c_{k}, f(X)\right) P\left(c_{k} | X\right) \end{aligned}$$为了使期望风险最小化，只需对$X=x$逐个极小化，由此得到:$$\begin{aligned} f(x) &=\arg \min _{y \in \mathcal{Y}} \sum_{k=1}^{K} L\left(c_{k}, y\right) P\left(c_{k} | X=x\right) =\arg \min _{y \in \mathcal{Y}} P\left(y \neq c_{k} | X=x\right) =\arg \min _{y \in \mathcal{Y}}\left(1-P\left(y=c_{k} | X=x\right)\right) =\arg \max _{y \in \mathcal{Y}} P\left(y=c_{k} | X=x\right) \end{aligned}$$这样一来，根据期望风险最小化准则就得到了后验概率最大化准则:$$f(x)=\arg \max _{c_{k}} P\left(c_{k} | X=x\right)$$

## 7. 算法实现
![](https://images2017.cnblogs.com/blog/1146184/201708/1146184-20170831162008405-131331666.png)
<center>朴素贝叶斯分类算法流程图</center>