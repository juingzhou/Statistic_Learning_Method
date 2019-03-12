# 感知器学习算法的对偶形式
- 相比于原始形式, 对偶形式在理解上没有那么直观，其基本思想就是将$ w $ 和 $ b $表示为实例$ x_i $ 和标记$ y_i $的线性组合的形式, 通过求解其系数进而求得$ w $和$b$。 
- 假设 $ w_0 = 0, b = 0$, 那么可以看出，当所有的点没有误判时,最后得到的$w,b$一定有如下的形式:
$$ (1)\ \ \begin{cases}
w = \sum_{i=1}^N n_i \eta y_i x_i = \sum_{i=1}^N \alpha_iy_ix_i \\
b = \sum_{i=1}^Nn_i \eta y_i = \sum_{i=1}^N \alpha_i y_i
\end{cases}$$
- 其中$\alpha_i = n_i \eta_i$中的$n_i$代表对第i样本的学习次数, 感知机对偶形式的完整形式为以下所示:$$ f(x) = sign(\sum_{j=1}^N \alpha_i y_i x_j x + b) \ \ \ \ \ (2)$$
- 具体步骤如下:
  1. 输入:线性可分的数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$，其中$x_i \in R^n,y_i \in {-1,+1},i=1,2,...,N;$学习率 $\eta (0 < \eta \le 1);$
  2. 输出:$\alpha,b;$ 感知机模型$f(x) = sign(\sum_{j=1}^N \alpha_i y_i x_j x + b) $
  其中 $\alpha=(\alpha_1,\alpha_2,...,\alpha_N)^T.$
      - (1) $ \alpha \leftarrow 0, b \leftarrow 0$
      - (2) 在训练集中选取数据$(x_i,y_i)$
      - (3) 如果$y_i\left(  \sum_{j=1}^N \alpha_jy_jx_jx_i + b \right) \le 0$ $$ \alpha_i \leftarrow \alpha_i + \eta \\
      b \leftarrow b + \eta y_i$$
      - (4) 转至$(2)$直到没有误分类数据
- 对偶形式中训练实例仅以内积的形式出现. 为了方便可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储, 这个矩阵就是所谓的$Gram$矩阵$(Gram \ matrix)$
$$G=[x_i \cdot x_j]_{N \times N}$$
  例如:
$$
\begin{bmatrix}x_1*x_1 & x_1*x_2 & \cdots  & x_1*x_N\\
x_2*x_1 & x_2*x_2 & \cdots &x_2*x_N \\
\vdots & \vdots & \ddots & \vdots \\ x_N*x_1 & x_N*x_2 & \cdots & x_N*x_N
\end{bmatrix}
$$
    