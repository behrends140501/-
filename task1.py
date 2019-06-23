
# coding: utf-8

# In[137]:


import numpy as np


# In[62]:


print(np.__version__)
np.show_config()


# In[63]:


z=np.zeros(10)
print(z.size,z.itemsize)
#np.info(np.add)


# In[64]:


a=np.arange(10,50)
a1=a[::-1]#反转向量
b=np.arange(9).reshape(3,3)
nz=np.nonzero([1,2,0,0,4,0])#非0索引
b1=np.eye(3)
b2=np.random.random((3,3,3))#创建一个3x3x3的随机数组
b2min,b2max=b2.min(),b2.max()
b3=np.random.random(30)
b3m=b3.mean()
z=np.ones((10,10))
z[1:-1,1:-1]=0#其中边界值1，其余值为0
c=np.ones((5,5))
c=np.pad(c,pad_width=1,mode='constant',constant_values=0)#用0填充边界
print(0*np.nan)#nan
print(np.nan == np.nan)#false
print(np.inf > np.nan)#false
print(np.nan - np.nan)#nan
print(0.3 == 3 * 0.1)#false
z=np.diag(1+np.arange(4),k=0)#对角线矩阵
Z = np.zeros((8,8),dtype=int)#棋盘状
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(np.unravel_index(100,(6,7,8)))
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))#8*8棋盘
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
color = np.dtype([("r", np.ubyte, 1),("g", np.ubyte, 1),("b", np.ubyte, 1),("a", np.ubyte, 1)])
x=np.dot(np.ones((5,3)),np.ones((3,2)))#实矩阵乘积
print(x)
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1#元素取反


# In[65]:


print(sum(range(5),-1))#求和是0、1、2、3、4、-1


# In[66]:


from numpy import *
print(sum(range(5),-1))#是在最后一维求和


# In[67]:


Z = np.arange(5)#生成整数向量
Z ** Z  # legal
2 << Z >> 2 # false


# In[68]:


A=np.arange(5)
2 << A >> 2 


# In[69]:


Z <- Z# legal
1j*Z#虚数
Z/1/1#？
Z<Z>Z#false


# In[70]:


print(np.array(0) / np.array(0))#nan
print(np.array(0) // np.array(0))#0
print(np.array([np.nan]).astype(int).astype(float))#？


# In[71]:


Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))#对浮点数做舍入


# In[72]:


Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))#找到共同元素


# In[75]:


# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0#忽略numpy警告


# In[76]:


np.sqrt(-1) == np.emath.sqrt(-1)#false


# In[77]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D') 
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday is " + str(yesterday))
print ("Today is " + str(today))
print ("Tomorrow is "+ str(tomorrow))


# In[78]:


Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)


# In[79]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)#？
np.multiply(A,B,out=A)


# In[80]:


Z = np.random.uniform(0,10,10)#五种取整方式
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))


# In[81]:


Z = np.zeros((5,5))
Z += np.arange(5)
print (Z)


# In[82]:


def generate():
    for x in range(10):
         yield x#生成器函数
Z = np.fromiter(generate(),dtype=float,count=-1)
print (Z)
Z = np.linspace(0,1,11,endpoint=False)[1:]
Z = np.random.random(10)
Z.sort()


# In[83]:


A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)#检查形状和元素值,可以用公差
equal = np.array_equal(A,B)#检查 


# In[84]:


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1#创建只读数组


# In[85]:


Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print (R)
print (T)#10*2矩阵转极坐标
Z = np.random.random(10)
Z[Z.argmax()] = 0#最大值替换成1
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))#结构化数值，并实现 x 和 y 坐标覆盖 [0,1]x[0,1] 区域
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)#给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))
print(np.linalg.det(C))


# In[86]:


for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# In[87]:


np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print (Z)#打印所有数值


# In[88]:


Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print (Z[index])#找到最接近标量的值


# In[89]:


Z = np.zeros(10, [ ('position', [ ('x', float, 1),('y', float, 1)]),('color',[ ('r', float, 1),('g', float, 1),('b', float, 1)])])
#创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组(★★☆)


# In[90]:


Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print (D) #对一个表示坐标形状为(100,2)的随机向量，找到点与点的距离
#方法二
import scipy
import scipy.spatial
D = scipy.spatial.distance.cdist(Z,Z)
print(D)


# In[91]:


Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
print (Z)#将浮点改成整数


# In[92]:


Z = np.arange(9).reshape(3,3)#枚举等价于numpy
for index, value in np.ndenumerate(Z):
    print (index, value)
for index in np.ndindex(Z.shape):
    print (index, Z[index])


# In[93]:


X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print (G)#生成高斯数组


# In[94]:


n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print (Z)#二维数组放置p个元素


# In[95]:


X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
print(Y）#减去一个矩阵中的每一行的平均值 
#法二
Y = X - X.mean(axis=1).reshape(-1, 1)
print (Y)


# In[96]:


Z = np.random.randint(0,10,(3,3))
print (Z)
print (Z[Z[:,1].argsort()])#对第n列对一个数组进行排序


# In[97]:


Z = np.random.randint(0,3,(3,10))
print ((~Z.any(axis=0)).any())#如何检查一个二维数组是否有空列？


# In[98]:


Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print (m)#从数组中给定值中找出最近的值


# In[103]:


A = np.arange(3).reshape(3,1)#用迭代器(iterator)计算两个分别具有形状(1,3)和(3,1)的数组? 
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: 
    z[...] = x + y
print (it.operands[2])


# In[104]:


class NamedArray(np.ndarray):
     def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
     def __array_finalize__(self, obj):
        if obj is None: return 
        self.info =getattr(obj,'name', "no name")
Z = NamedArray(np.arange(10),"range_10")
print (Z.name)#创建一个具有name属性的数组类


# In[105]:


Z = np.ones(10)#第二个向量索引的每个元素加1
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)
#法二
np.add.at(Z, I, 1)
print(Z)


# In[108]:


X = [1,2,3,4,5,6]#根据索引列表(I)，将向量(X)的元素累加到数组(F)
I = [1,3,9,3,4,1] 
F = np.bincount(I,X)
print (F)


# In[109]:


w,h = 16,16#考虑一个(dtype=ubyte) 的 (w,h,3)图像，计算其唯一颜色的数量
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*(256*256) + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print (n)


# In[112]:


A = np.random.randint(0,10,(3,4,3,4))#考虑一个四维数组，如何一次性计算出最后两个轴(axis)的和？
sum = A.sum(axis=(-2,-1))
print (sum)
#方法2
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print (sum)


# In[113]:


A = np.random.uniform(0,1,(5,5))#获得点积的对角线
B = np.random.uniform(0,1,(5,5))
np.diag(np.dot(A, B))
#法二
np.sum(A * B.T, axis=1)
#法三
np.einsum("ij,ji->i", A, B)


# In[114]:


Z = np.array([1,2,3,4,5])#一个向量[1,2,3,4,5]，建立一个新的向量，在这个新向量中每个值之间有3个连续的零
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print (Z0)


# In[115]:


A = np.ones((5,5,3))#维度(5,5,3)的数组，将其与一个(5,5)的数组相乘
B = 2*np.ones((5,5))
print (A * B[:,:,None])


# In[116]:


A = np.arange(25).reshape(5,5)#个数组中任意两行做交换? 
A[[0,1]] = A[[1,0]]
print (A)


# In[117]:


faces = np.random.randint(0,100,(10,3))#
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print (G)


# In[118]:


C = np.bincount([1,1,2,3,4,4,6])#产生一个数组A满足np.bincount(A)==C
A = np.repeat(np.arange(len(C)), C)
print (A)


# In[121]:


def moving_average(a, n=3): #通过滑动窗口计算一个数组的平均数
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)

print(moving_average(Z, n=3))


# In[122]:


Z = np.random.randint(0,2,100)#对布尔值取反，或者原位(in-place)改变浮点数的符号(sign)
np.logical_not(Z, out=Z)


# In[143]:


#计算每一个点 j(P[j]) 到每一条线 i (P0[i],P1[i])的距离
from scipy.spatial.distance import *
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print (np.array([scipy.spatial.distance(P0,P1,p_i) for p_i in p]))


# In[144]:


#考虑一个数组Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],如何生成一个数组R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ...,[11,12,13,14]]? 
Z = np.arange(1,15,dtype=np.uint32)
R = np.lib.stride_tricks.as_strided(Z,(11,4),(4,4))
print (R)


# In[135]:


#矩阵的秩
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print (rank)


# In[134]:


#频率最高的值
Z = np.random.randint(0,10,50)
print (np.bincount(Z).argmax())


# In[139]:


#从一个10x10的矩阵中提取出连续的3x3区块
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print (C)


# In[128]:


class Symetric(np.ndarray):#创建一个满足Z[i,j] == Z[j,i]的子类 
        def __setitem__(self, index, value):
            i,j = index
            super(Symetric, self).__setitem__((i,j), value)
            super(Symetric, self).__setitem__((j,i), value)
def symetric(Z):
     return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print (S)


# In[129]:


Z = np.arange(10000)#找到一个数组的第n个最大值
np.random.shuffle(Z)
n = 5
print (Z[np.argsort(Z)[-n:]])
#法二
print (Z[np.argpartition(-Z,n)[:n]])


# In[132]:


A = np.random.randint(0,5,(8,3))#在数组A中找到满足包含B中元素的行
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print (rows)


# In[131]:


X = np.random.randn(100)#计算它boostrapped之后的95%置信区间的平均值 计算它boostrapped之后的95%置信区间的平均值
N = 1000 
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print (confint)

