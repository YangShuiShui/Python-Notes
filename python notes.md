Python notes

<center>@author: Qingyuan Yang <br>

# Reference

- Python for data analysis (2nd edition) 
- 10 minutes to pandas 0.25.3
- Think python (2nd edition)
- Effective python (2nd edition)

# Preliminaries

## What kinds of data?

主要关注结构化数据(structured data)

- 表数据 Tabular and spreadsheet-like data
- 多维数组/矩阵 Multidimensional arrays (matrices)
- 键列关联的多个数据表 / 关系数据库 SQL
- 均匀或不均匀间隔的时间序列 Evenly or unevenly spaced time series
- And so on

非结构化数据需要提取特征features，找寻其中的结构化的部分

## Strength and weakness of Python

- **Strength 1: Solving the 'two-language' problem**

  测试和实盘都可以用python，python既可以做研究，也可以构建系统 (但好像构建实盘的交易系统还是C++用的更多)

- **Strength 2: Python as glue**

  可以调用别的语言，用起来更方便灵活。Python博览百家之长？

- **Weakness 1: Slower than complied language**

  解释性语言运行速度往往低于编译语言(Java/C++)，trade-off between programmer time and CPU time. 当工作环境有运算速率的需求时(高频交易系统)，还是C++这些更好。

- **Weakness 2: Not suitable for highly concurrent, multithreaded applications**

  多线程的程序Python表现较为乏力，由于存在全局解释器锁 global interpreter lock (GIL)

## Essential Python libraries

- **Numpy**
- **pandas**
- **matplotlib**
- **IPython**
- **SciPy** 科学计算包
  - scipy.integrate: 数值积分和
  - scipy.linalg: 线性代数
  - scipy.optimize: 函数优化器与根查找算法
  - scipy.signal: 信号处理工具
  - scipy.sparse: 稀疏矩阵和稀疏线性系统求解器
  - scipy.special: 常用数学函数
  - scipy.stats: 数理统计与概率论
- **scikit-learn** 机器学习包
  - Classifications: SVM, nearest neighbors, random forest, logistic regression, etc.
  - Regression: Lasso, ridge regression, etc.
  - Clustering: k-means, spectral clustering, etc.
  - Dimensionality reduction: PCA, feature selection, matrix factorization, etc.
  - Model selection: Grid search, cross-validation, metrics
  - Preprocessing: Feature extraction, normalization
- **statsmodels** 古典统计与计量包
  - Regression models: Linear regression, generalized linear models, robust linear models, linear mixed effects models, etc.
  - Analysis of variance (ANOVA)
  - Time series analysis: AR, ARMA, ARIMA, VAR, and other models
  - Nonparametric methods: Kernel density estimation, kernel regression
  - Visualization of statistical model results

## Navigating

- **Interacting with the outside world**

  Reading and writing with a variety of file formats and data stores

- **Preparation**

  Cleaning, munging, combining. normalizing, reshaping, slicing and dicing, and transforming data for analysis

- **Transformation**

  Applying mathematical and statistical operations to groups of datasets to derive new datasets (e.g., aggregating a large table by group variables)

- **Modeling and computation**

  Connecting your data to statistical models, machine learning algorithms, or other computational tools

- **Presentation**

  Creating interactive or static graphical visualizations or textual summaries

## Import conventions and jargon

Some import conventions:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sm
```

Jargon:

- Munge/munging/wrangling 数据规整
- Pseudocode 伪代码
- Syntactic sugar 语法糖

## 编码风格

**空格**

- 每行的字符数不要超过79；
- 函数与类之间用两个空行隔开；
- 同一个类中，各方法用一个空行隔开；
- 在使用下标获取列表元素、调用函数或给关键字参数赋值的时候，不要在两边加空格；
- 为变量赋值的时候，左右加空格；

**命名**

- 函数(function)、变量(variable)、属性(attribute)应该用小写字母来拼写，下划线连接单词；
- 受保护的实例属性，应该以单个下划线开头；
- 私有的实例属性，应该以两个下划线开头；
- 类和异常，应该每个单词首字母均大写；
- 模块级别的常量，应该全部采用大写字母，单词用下划线连接，如ALL_CAPS；
- 类中的实例方法(instance method)，应该把首个参数设置为self，以表示对象本身；
- 类方法(class method)的首个参数，应该命名为cls，以表示该类自身；

**模块**

- 最上方写标准库模块；
- 中间写第三方模块；
- 最下方写自用模块；

![image-20200812154335746](python%20notes.assets/image-20200812154335746.png)

# Python basic

## Setting path和os库

```python
# 设置路径
import os
os.chdir(r'F:\balabala\balala')
import sys;sys.path.append(r'\路径') # 一般导入自写模块用这个
# sys用多了会有一个bug，很多路径里有同名文件，不知道该调用哪个 不推荐频繁变更路径的情况下使用

# 查看当前路径
os.getcwd() # current working directory

# 查看是否存在文件
os.path.exists('test.txt')

# 查看路径下的文件名
os.listdir()

# 遍历路径下的文件、路径
for root, dirs, files in os.walk(r'D:\我爱读书！'):
    print(root)
    print(dirs)
    print('{}\n'.format(files))
    
# 拼接路径和文件名
os.path.join()

# 导入模块的顺序

# gotopath
def gotopath(path):
    '''
    If the path doesn't exist, create one and get to.
    Otherwise, get to the existing path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        os.chdir(path)
    else:
        os.chdir(path)
```

## Import modules

```python
# installation
win + R --> cmd --> d: --> pip install *.whl
## 输入*的时候，只需输入最前面的几个字母，再按tab补全

# installation
pip/conda install -i https://pypi.tuna.tsinghua.edu.cn/simple + module # 用清华下速度快
    
# 间接访问module定义的变量和函数
import some_module
some_module.func() # 调用函数需要加some_module
a = some_module.a # 调用变量也要加some_module

# 直接访问module部分定义的变量与函数
from some_module import a, func
print(a)
func(a)

# 也可以重新命名哦
import some_module as sm
sm.func(a)
a = sm.a

import some_module as sm
from some_module import PI as pi, g as gf
r1 = sm.f(pi)
r2 = gf(6, pi)
```

## Types and attributions

```python
# 查看对象属性
dir()

# 访问对象属性
getattr(a, 'attribute')
hasattr()
setattr()

# type ----------------------------------------------------
# 查看类型
type(a) # 查看a的类
a.dtype # 查看a的数据类型

# 判断对象类型
isinstance(a, int)
isinstance(a, (int, float))

# 转换类型
str(a)
bool(0) # False
bool(1) # True
int(a)
int(True) # 1
int(False) # 0
float(a)
tuple(a)
df.astype(float) # 

# 将dataframe中的布尔值全部转换成0,1
for u in df.columns:
    if df[u].dtype == bool:
        df[u] = df[u].astype('int')
```

## Data structures

### String

```python
# r'balabala'称为原始字符串，引号里忽略所有转义字符
3 * 'ba' + 'lala' # return 'bababalala'
'abcd'[2] # return 'c'
'abcd'[-1] # return 'd'
'abcd'[:2] # return 'ab'

# 万物皆可字符串
str()

# 字符串转成list处理也很香哦
s = 'python'
list(s)

# 查找与替换
str.replace('abc','') 

# lower/upper
str.capitalize() # 首字母大写
str.title() # 所有单词首字母大写
str.casefold() # 全部变成小写
str.lower() 
str.upper()
str.swapcase() # 大小写互换

# arrange
str.center(width,fillchar) # 居中对齐，占位width，左右用fillchar填充
str.rjust(width,fillchar) # 靠右对齐

# count
len('abcd') # return 4
str.count(sub) # str中sub出现的次数

# find
str.find('*') # 返回最小索引 "第一次在哪里发现它的？"
str.rfind('*') # 返回最大索引，right find 从右边开始找

# True or false?
str.startswith(char)
str.endswith('*') # 返回true or false 是否以*结尾
str.isalnum() # 是否都是字母和数字 不能有其他字符
str.isalpha() # 是否只有字母
str.isdecimal() # 是否只有十进制数
str.isdigit() # 是否是数值型
str.isidentifier() # 是否是保留字 True False def return等等
str.islower() # 是否全小写
str.isupper() # 是否全大写
str.isnumeric() # 是否数值型

# strip
str.lstrip(char) # 移除字符串最前面的char中存在的字符
str.rstrip(char) # 从右开始，同上
str.strip(char) # 左右开弓，同上

# split
str.rpartition(char) # 从右边开始找char，找到后分割成三段
str.split(sep=char,[maxslpit=*]) # 从左开始，无限分割/*次分割
str.rsplit() # 从右开始 同上

# join
'.'.join(['a','b','c']) # return  a.b.c
str.join() # 用str来连接列表
```

![image-20200606094336051](python%20notes.assets/image-20200606094336051.png)

![image-20200606094423784](python%20notes.assets/image-20200606094423784.png)

### 格式化输出

```python
# 自动用0补齐
print('{:02d}:{:02d}:{:02d}'.format(hour, minute, second)) # 12:00:00

# 千分位分隔符
print('{:,}'.format(12345)) # 12,345
```

### List

```python
# create
list1 = [1,2,3]

# loc 切片
## 列表切片的一些建议
'''
尽量不要同时使用stride、start、end，会让人费解；如果需要，分成两步操作；
尽量不要用负数的stride
'''
list1[0] # = 1
list1[-1] # = 3
list1[-2:] # = [2,3]
list1[0] = 10 # 给第一位赋值
a[::-1] # 将列表倒序
list(reversed(range(10))) # 从后往前迭代 
### reversed是一个生成器，只有实体化（列表或for循环）才能创建序列
a[::2] # step=2

# concat
list2 = [4,5,6]
list1 + list2 # = [1,2,3,4,5,6]
list1.extend(list2)
list1.append(4) # = [1,2,3,4]
list.insert(1,'red') # 在1位置添加'red'
list.insert(i,x) # 在i处插入x

# embed
list3 = [list1,list2]
list3[1] # = list2
list3[1][1] # = list2[2] = 5	

# remove
list.remove(x) # 移除 "第一个" 值为x的数
list.pop(index) # 移除第i个值，并返回它，公开处刑
list.clear() # drop all

# count
list.count(x) # how many times does x appear in list?

# sort / sorted
a.sort(reverse=False) # 在原列表基础上直接改
a.sort(key=len) # 按照长度排序
a2 = sorted(a) # 原列表保留，创建新列表
sorted('house race') # 直接对字符串进行排序

# copy
a.copy()
a[:]

# in
list1 = [1,2,3,4]
2 in list1 # True
5 not in list1 # True

# bisect
import bisect
bisect.bisect(list1,10) # 对给定排序的list1，找到10应该插入的位置
bisect.insort(list1,10) # 将10插入已排序的list1

# list删除重复项
list1 = ['123','321','123']
list_drop = list(set(list1))
```

### Tuple

```python
# 应用场景
## 迭代元祖或列表序列
seq = [(1,2,3),(3,4,5),(5,6,7)]
for a,b,c in seq:
    balabala
    
# 赋值
tup = 4,5,6
tup = 4,5,(6,7)
tup = 4,5,(6,7),'hhh'
tup = (4,5),(7,8)

# 切片
tup[1:]
tup[2]

# 操作
('a','b') * 4 # 复制串联
tup[1].append(3) # 原位修改

# 元组拆分
values = 1,2,3,4,5
a,b,*rest = values
a,b,*_ = values # 习惯性把不要的值放在_里

# 方法
a = (1,2,2,4,5)
a.count(2) # a中有几个2

# 元组比大小
(1,2) < (2,2) # 先比较第一位，如果小于则返回True；如果相等，再比较第二位
```

### Sequence functions

```python
# 迭代序列时，跟踪序号
for i,value in enumerate(collection, i):
    # do something with value
# tqltql....!!!
# i是序号，value是值

'''可以给enumerate提供第2个参数，指定开始计数时所用的值'''
```

### Multi-dimensional data (Zip)

```python
seq1 = ['foo','bar','baz']
seq2 = ['one','two','three']
zipped = zip(seq1,seq2)
list(zipped)
# zip可以处理任意多的序列，元素的个数取决于最短的序列
seq3 = [False,True]
list(zip(seq1,seq2,seq3))

# zip的常见用法之一是同时迭代多个序列，可能结合enumerate使用
for i, (a, b) in enumerate(zip(seq1, seq2)):
     print('{0}: {1}, {2}'.format(i, a, b))
        
# 解压多维数组 zip(*zipped)
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),('Schilling', 'Curt')]
temp1,temp2 = zip(*pitchers)

# 用zip遍历两个迭代器
for a, b in zip(seq1, seq2):
    print(a,b)
```

### Dict (hash map / associative array)

```python
# 创建、删除、更新 
d1 = {'a' : '123', 'b' : [1,2,3]} # 键值对,可以解读成 键就是index 值就是index对应的值
del d1['a']
d1.pop('a') # 删除a，返回'123'
d1.update({'a':'234','c':'hhh'})

d = {}
d[tuple([1,2,3])] = 5
d['a'] = 3
# 键值操作
'a' in d1 # True
d1.keys() # list[d1.keys()]
d1.values()

# 用序列创建字典
mapping = {}
for key,value in zip(key_list,value_list):
    mapping[key] = value

mapping2 = dict(zip(key_list,value_list))

dict comprehensions # 另一种构建字典的优雅方式

# 默认值 default
if key in some_dict:
    value = some_dict[key]
else:
    value = default_value
    
value = some_dict.get(key, default_value) # 如果key在some_dict的键中，返回这个字典中key对应的值，否则，返回默认值

## solution 1: for循环
words = ['apple','bat','bar','atom','book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
## solution 2: setdefault
for word in words:
    letter = word[0]
    by_letter.setdefault(letter,[]).append(word)
    # 如果dict(by_letter)中不包含key(letter)，则在dict中添加key，并设默认值[]，最后返回key对应的值；如果包含，则直接返回key对应的值
## solution 3: defaultdict
from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
```

### Set

```python
# 使用场景
## 就是数学里的集合啊！！！不会重复的那种啊！！！
## 并集、交集、差分、对称差balabala......

# 创建
s = set([1,2,2,3])
s = {1,2,2,3}

# 方法
a.union(b) # 取并集
a|b # 取并集
a.intersection(b) # 取交集
a&b # 取交集
```

![img](python%20notes.assets/7178691-980efe5d98ecc4d6.png)

### List, set and dict comprehensions 列表、字典、集合推导

```python
## 建议
'''
列表推导支持多级循环，每层循环也支持多项条件；
超过2个表达式的列表推导很难理解，应该尽量避免。
'''
# List comprehensions
[expression for val in collection if condition]
## equal to
result = []
for val in collection:
    if condition:
        result.append(expr)
## for example        
a = [1,2,3,4,5]
[i*2 for i in a if i>2]

## 嵌套列表推导式 nested list comprehensions
all_data = [['john','Emily','Michael','Mary','Steven'],
            ['Maria','Juan','Javier','Natalia','Pilar']]
results = [name for names in all_data for name in names if name.count('e')>=2]

some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for i in some_tuples for x in i]
[[x for x in tup] for tup in some_tuples]

# dict comprehensions
dict_comp = {key-expr : value-expr for value in collection if condition}
{i*2 for i in a if i>2}

loc_mapping = {val:index for index,val in enumerate(strings)}

# set comprehensions
set_comp = {expr for value in collection if condition}
set(map(len,string)) # 将string与其长度map上
def half(x):
    x = x / 2
    return x
set(map(half,a))

# map函数
map(func,object) # 对object进行func处理 
```

### 生成器和迭代器

```python
'''
当输入的数据量较大时，列表推导可能会因为占用太多内存而出问题。
由生成器表达式所返回的迭代器，可以逐次产生输出值，从而避免了内存用量问题。
把某个生成器表达式所返回的迭代器，放在另一个生成器表达式的for子表达式中，即可将二者组合起来。
串在一起的生成器表达式执行速度很快。
'''
value = [len(x) for x in open('/tmp/my_file.txt')]
print(value)

# 转化为生成器表达式，对生成器表达式求值的时候，它会立刻返回一个迭代器。
it = (len(x) for x in open('/tmp/my_file.txt'))
print(it)
print(next(it))

def index_word(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1
```

### Class 类和继承

类的第一个例子：Time

```python
class Time:
    '''Represents the time of the day.'''
    def __init__(self, hour=0, minute=0, second=0): # 初始赋值 time = Time(1,2,3)
        self.hour = hour
        self.minute = minute
        self.second = second
    
    def __str__(self): # print(time)的默认输出方法
        return '{:02d}:{:02d}:{:02d}'.format(self.hour, self.minute, self.second)
    
    def print_time(self):
        print('{:02d}:{:02d}:{:02d}'.format(self.hour, self.minute, self.second))

    def time_to_int(self):
        minutes = self.hour * 60 + self.minute
        seconds = minutes * 60 + self.second
        return seconds
    
    def int_to_time(seconds):
        time = Time()
        minutes, time.second = divmod(seconds, 60)
        time.hour, time.minute = divmod(minutes, 60)
        return time     

    def __add__(self, other): # 允许加法
        if isinstance(other, Time):
            return self.add_time(other)
        else:
            return self.increment(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def add_time(self, other):
        seconds = self.time_to_int() + other.time_to_int()
        return int_to_time(seconds)
    
    def increment(self, seconds):
        seconds += self.time_to_int()
        return int_to_time(seconds)   
```

类的第二个例子：一张牌、一副牌、一手牌

```python
class Card:
    '''Represents a standard playing card.'''
    def __init__(self, suit=0, rank=2): # suit表示花色，rank表示点数
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        suit_name = ['梅花','方块','红桃','黑桃']
        rank_name = [None,'A','2','3','4','5','6','7','8','9','10','J','Q','K']
        return '{}{}'.format(suit_name[self.suit], rank_name[self.rank])
    
    def __lt__(self, other): # less than
        if self.rank < other.rank:
            return True
        else:
            return False
        
class Deck:
    def __init__(self):
        self.cards = []
        for suit in range(4):
            for rank in range(1,14):
                card = Card(suit, rank)
                self.cards.append(card)
    
    def __str__(self):
        cards = []
        for card in self.cards:
            cards.append(str(card))
        return '\n'.join(cards)
    
    def pop_card(self):
        return self.cards.pop() # 拿出最后一张牌
    
    def add_card(self, card):
        self.cards.append(card) # 在末尾添加一张牌
        
    def shuffle(self): # 洗牌
        random.shuffle(self.cards)
        
class Hands(Deck): # Hands可以直接继承Deck拥有的方法
    pass 
```

辅助类

```python
'''
不要使用包含其他字典的字典，不要使用过长的元组；
如果容器中包含简单而又不可变的数据，可以用namedtuple，之后再修改成完整的类；
保存内部状态的字典如果变得比较复杂，就应该把代码拆解成多个辅助类
'''
import collections
Grade = collections.namedtuple('Grade', ('score', 'weight'))

class Subject(object): # 辅助类
    def __init__(self):
        self._grades = []
    
    def report_grade(self, score, weight):
        self._grades.append(Grade(score, weight))
    
    def average_grade(self):
        total, total_weight = 0, 0
        for grade in self._grades:
            total += grade.score * grade.weight
            total_weight += grade.weight
        return total / total_weight
    
class Student(object): # 辅助类
    def __init__(self):
        self._subjects = {}
        
    def subject(self, name):
        if name not in self._subjects: # 键中是否有name
            self._subjects[name] = Subject()
        return self._subjects[name]
    
    def average_grade(self):
        total, count = 0, 0
        for subject in self._subjects.values():
            total += subject.average_grade()
            count += 1
        return total / count
    
class Gradebook(object):
    def __init__(self):
        self._students = {}
    
    def student(self, name):
        if name not in self._students:
            self._students[name] = Student()
        return self._students[name]
    
book = Gradebook()
albert = book.student('albert')
math = albert.subject('math')
math.report_grade(90,0.3)
math.report_grade(80,0.6)
chinese = albert.subject('chinese')
chinese.report_grade(100,0.5)
print(albert.average_grade())
```

## Input

```python
x = int(input("Please input an integer:"))
# input接受到的值 是'print'，带引号的，务必注意 都是string
```

## Control phases

```python
# while
while condition:
	order

# if
if condition1:
    order1
elif condition2:
    order2
elif condition3:
    order3
else:
    order4
    
'Non-negative' if x >= 0 else 'Negative' # 三元表达式
    
# for + list/range/series
for i in list/range()/Series:
    order    
range(start,end,step) # [start,end)
range(len)
for i in range(len(listA)):
    print(i,listA[i])

# for + else
for a in range(10):
    print(a)
else: # 循环穷尽/False时执行else
    print('Done')
    
# break 跳出最近的for和while循环
for n in range(2, 10):
     for x in range(2, n):
         if n % x == 0:
             print(n, 'equals', x, '*', n//x)
             break
     else:
         # loop fell through without finding a factor
         print(n, 'is a prime number')

# continue 忽略之后的步骤，直接进入循环中下一迭代
for num in range(2, 10):
     if num % 2 == 0:
         print("Found an even number", num)
         continue
     print("Found a number", num)
```

## Functions

```python
# python的函数是一个指针，更像是一种过程而非一个函数
def funcname():
    procession
    return value/list/None

# 参数默认值 default paramaters
def func(par1,par2=10,par3='method1'):
    procession
    return balabala

def func(par1,par2=10,par3=None): # 用none作为默认参数
    result = a + b
    if c is not None:
        result = result * c
    return result

# 关键字参数 keyword **kwarg = value
def func(par1,*arg,**kwarg):
    procession
    return balabala
func(1,2,3,4,a=5,b=6) # par1=1,*arg=(2,3,4),**kwarg={'a':5,'b':6}

# 哦对了 func可以返回多个值
return a,b,c # 一个元组
return_value = f() # return_value[0/1/2]调用a,b,c
return {'a':a, 'b':b, 'c':c}
```

### Namespaces, scope, and local functions

```python
# 函数可以访问两种不同作用域中的变量：global和local，python有一种更科学的用于描述变量作用域的名称，即命名空间namespace。任何在函数中赋值的变量默认都是被分配到局部命名空间（local namespace）中的。局部命名空间是在函数被调用时创建的，函数参数会立即填入该命名空间。在函数执行完毕之后，局部命名空间就会被销毁。

def func():
    a = []
    for i in range(5):
        a.append(i)
func() # 什么都不会发生，因为在local namespace创建了a，又删除了

a = []
def func():
    for i in range(5):
        a.append(i)
func() # a=[0,1,2,3,4] 函数访问access了global namespace并找到了a，并对a进行操作

def func():
    for i in range(5):
        a.append(i)
func() # 报错，访问不到a

def func():
    global a
    a = []
func() # 会在global namespace访问a或创建a 并对a操作

a = 1
def func():
    a = 0 # 在函数内部命名空间里创建了一个a
func() # a = 1

a = 1
def func():
    nonlocal a # nonlocal不能追溯到模块
    a = 0
func() # a = 0
'''
python解释器将按如下顺序遍历各作用域：
-当前函数的作用域
-任何外围的作用域（例如，包含当前函数的其他函数）
-包含当前代码的那个模块的作用域（global 全局作用域）
-内置作用域（包含len及str等函数的那个作用域）
'''
```

### Functions are objects!

```python
# method 1
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda','south   carolina##', 'West virginia?']
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]','',value)
        value = value.title()
        result.append(value)
    return result
clean_strings(states)

# method 2
def clean(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip, clean, str.title]

def clean_strings(strings,ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops)
```

### Anonymous (lambda) functions

```python
# lambda x: 后的结果就是返回值
# lambda函数之所以会被称为匿名函数，与def声明的函数不同，原因之一就是这种函数对象本身是没有提供名称name属性。
## 习惯性用法
f = lambda x: x * 2
f(2)

def short_function(x):
    return x * 2
equiv_anon = lambda x: x * 2

strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
```

### Currying: partial argument application

```python
# 柯里化（currying）是一个有趣的计算机科学术语，它指的是通过“部分参数应用”（partial argument application）从现有函数派生出新函数的技术。
def add_numbers(x, y):
    return x + y
add_five = lambda y: add_numbers(5, y)

from functools import partial
add_five = partial(add_numbers, 5)
```

### Generators

```python
# 迭代器协议iterator protocol：以一种一致的方式对序列进行迭代 iter
some_dict = {'a':1,'b':2,'c':3}
for key in some_dict:
    print(key)
## 上述代码实际上是尝试从some_dict创建一个迭代器
dict_iterator = iter(some_dict)
'''迭代器是一种特殊对象，它可以在诸如for循环之类的上下文中向Python解释器输送对象。大部分能接受列表之类的对象的方法也都可以接受任何可迭代对象。比如min、max、sum等内置方法以及list、tuple等类型构造器'''
min(dict_iterator); max(dict_iterator); list(dict_iterator)
## 迭代器用一次少一次......使用过的就丢掉了

# 生成器
'''
生成器（generator）是构造新的可迭代对象的一种简单方式。一般的函数执行之后只会返回单个值，而生成器则是以延迟的方式返回一个值序列，即每返回一个值之后暂停，直到下一个值被请求时再继续。
'''
## 生成器也是用一回少一回...迭代过的数字就会丢掉了
def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n**2))
    for i in range(1, n+1):
        yield i ** 2
gen = squares() # 不会有任何东西生成
for x in gen:
    print(x,end=' ') # 从生成器中请求元素时，才会执行代码

## 生成器表达式
gen = (x ** 2 for x in range(100))
sum(x ** 2 for x in range(100))
dict((i,i**2) for i in range(5))

# itertools module
## itertools模块有一组用于许多常见数据算法的生成器
import itertools
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator
```

![img](python%20notes.assets/7178691-111823d8767a104d.png)

### 纯函数和修改器

python允许**纯函数**和**修改器**两种形式的function。

**纯函数**就是输入参数、返回值，**对原参数不进行修改**。

```python
# 原本的t1, t2不会发生改变，新生成了一个sum1的函数
def add_time(t1, t2):
    sum1 = Time()
    sum1.hour = t1.hour + t2.hour
    sum1.minute = t1.minute + t2.minute
    sum1.second = t1.second + t2.second   
    if sum1.second >= 60:
        sum1.second -= 60
        sum1.minute += 1
    if sum1.minute >= 60:
        sum1.minute -= 60
        sum1.hour += 1
    return sum1
```

而**修改器**是直接进行操作，操作过程中**函数外的参数也会发生变化**。修改器可以理解为一个**步骤执行器**。

```python
# time的值直接被修改了
def increment(time, seconds):
    time.second += seconds
    if time.second >= 60:
        time.second -= 60
        time.minute += 1
    if time.minute >= 60:
        time.minute -= 60
        time.hour += 1
```

一般而言，我们尽量用纯函数形式，而在main中可以使用修改器形式。

## Errors and exception handling

```python
# 只要try不对，就执行except
try:
    float(x)
except:
    return x

# 当try犯ValueError错误时，执行except
try:
    float(x)
except ValueError: 
    return x

# 当try犯ValueError或TypeError时，执行except
try:
    float(x)
except (TypeError, ValueError): 
    return x

# 不管报不报错，都想执行一段代码
try:
    float(x)
finally:
    print('Don\'t worry!')
    
# 只有try成功的时候，才执行else部分代码
try:
    float(x)
except:
    print('Failed')
else:
    print('Succeeded')
finally:
    print('You are so good!')
    
'''else的主要作用是让try足够简短，except、else分别表示try成功失败的情况'''

assert something, 'balabala' # 判断something，如果是错的，则报错AssertError: balabala
```

# Numpy

**部分功能**

- ndarray，一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。
- 用于对整组数据进行快速运算的标准数学函数（无需编写循环）。
- 用于读写磁盘数据的工具以及用于操作内存映射文件的工具。
- 线性代数、随机数生成以及傅里叶变换功能。
- 用于集成由C、C++、Fortran等语言编写的代码的A C API。

**主要应用场景**

- 用于数据整理和清理、子集构造和过滤、转换等快速的矢量化数组运算。
- 常用的数组算法，如排序、唯一化、集合运算等。
- 高效的描述统计和数据聚合/摘要运算。
- 用于异构数据集的合并/连接运算的数据对齐和关系型数据运算。
- 将条件逻辑表述为数组表达式（而不是带有if-elif-else分支的循环）。
- 数据的分组运算（聚合、转换、函数应用等）。

## ndarray: a multidimensional array object

### Creating ndarrays

![img](python%20notes.assets/7178691-78ab11f67e7077a6.png)

```python
# short for numerical python
import numpy as np
data = np.random.rand(2,3)
data * 10
data + data

# ndarray是一个通用的同构数据多维容器，所有元素必须相同类型
data.shape # 必须相等
data.dtype # 必须相等

# 创建ndarray
'''
4种方式
	1. np.array 序列转换成ndarray
	2. np.zeros 全0数组
	3. np.ones 全1数组
	4. np.empty 全空数组
	5. np.arange 类似range函数
'''
data1 = [1,1,2,3,4,5]
arr1 = np.array(data1)
arr1.ndim # =1
arr1.shape # (5,)
arr1.dtype # float64

data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2.ndim # =2 两个维度,index * column
arr2.shape # (2,4)
arr2.dtype # int32

data3 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
arr3 = np.array(data3)
arr3.ndim # =2
arr3.shape # (3,4)

arr4 = np.zeros((10,2)) # 换成np.ones / np.empty一样
arr5 = np.zeros((3,4,5)) 

# 生成随机数
np.random.randint(min,max,length)
np.random.randn(d1,d2,d3,...)

# 改变形状
arr.reshape((4,2),order='C'/'F') # C按照行排列，F按照列排列 
arr.reshape((4,-1)) # 维数=-1表明该维度的大小由数据本身计算得来，爷不算了，机器算去
arr.ravel() # 扁平化 变成一行
arr.flatten() # 变成一行 跟上面一样？

# 合并和拆分
np.concatenate([arr1,arr2],axis=0) # 类似pd.concat([],axis=0)
np.vstack((arr1, arr2)) # 上下拼 vertical垂直拼
np.hstack((arr1, arr2)) # 左右拼 horizontal水平拼

first, second, third = np.split(arr, [1, 3]) # 在第二行 第四行上方切一刀

# r_和c_
np.r_[arr1, arr2] # 按行拼接，上下拼
np.c_[arr1, arr2] # 按列拼接，左右拼
np.c_[1:6, -10:-5] # 转化成数组

# 重复repeat、tile
arr = np.arange(3) # 0,1,2
arr.repeat(3) # 0,0,0,1,1,1,2,2,2
arr * 3 # 0,3,6
arr.repeat([1,2,3]) # 0,1,1,2,2,2
# 多维数组
arr = np.random.randn(2, 2)
arr.repeat([2, 3], axis=0) 
np.tile(arr, (2,3)) # 2行，3列

# 索引与替换
inds = [3,4,7,1] # index
arr.take(inds) # 取第3,4,7,1个元素
arr.take(inds, axis=0) # 取第3,4,7,1行
arr.take(inds, axis=1) # 取第3,4,7,1列
arr.put(inds, [40, 41, 42, 43]) # 将3、4、7、1用40,s41,42,43代替
```

![image-20200608154831497](python%20notes.assets/image-20200608154831497.png)

### Data type for ndarrays

![img](python%20notes.assets/7178691-2f2d7406a8bc076c.png)

![img](python%20notes.assets/7178691-5cc31115615737b7.png)

```python
arr1 = np.array([1,2,3], dtype=np.float64)
arr1 = arr1.astype(np.int32)
arr1.dtype
arr2 = arr1.astype(arr1.dtype)
```

### Arithmetic and slice

```python
'''
数组很重要，因为它使你不用编写循环即可对数据执行批量运算。NumPy用户称其为矢量化（vectorization）。大小相等的数组之间的任何算术运算都会将运算应用到元素级：
'''
arr1 + arr2
arr1 * arr2
arr1 * 2
arr1 ** 2
arr1 / 2
arr1 > arr2
'''
不同大小的数组之间的运算叫做广播（broadcasting）
'''
# 切片是一种view操作，并没有创建出新的数组
arr = np.arange(10) 
arr[5:8] = 10
arr_slice = arr[5:8]
arr_slice = 10 # 原数组依然会发生变化
data[Bool]
data[data2 == 'hhh',2:]
data[data2 != 'hhh']
data[~(data2 == 'hhh')]
```

## Universal Functions: fast element-wise array functions

![img](python%20notes.assets/7178691-1d494e73b61c7ced.png)

![img](python%20notes.assets/7178691-4e38d02a66481530.png)

![img](python%20notes.assets/7178691-eff1e61e5464159f.png)

```python
arr = np.arange(10)
np.sqrt(arr)
np.sqrt(arr,arr2) # 将arr的sqrt返回给arr2
np.exp(arr)
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x,y)
```

## 利用数组进行数据处理

可以通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算。sum、mean以及标准差std等聚合计算（aggregation，通常叫做约简（reduction））既可以当做数组的实例方法调用，也可以当做顶级NumPy函数使用。

```python
# 计算sqrt(x^2 + y^2)
points = np.arange(-5,5,0.01)
xs, ys = np.meshgrid(points,points)

# np.where
np.where(arr > 0, 2, arr) # 正数用2替换，其余不变 替换值
```

![img](python%20notes.assets/7178691-a6c6df3ca8e0b98e.png)![img](python%20notes.assets/7178691-866fcde885b1d357.png)

```python
arr = np.random.randn(5,4)

# mean/sum可以传入axis为控制轴的参数，会返回一个少一维度的结果
arr.sum(axis=1) # 计算行
arr.mean(axis=0) # 计算列

# 用于布尔型数组的计算
arr = np.random.randn(100)
(arr>0).sum()

# 测试数组中是否存在一个或多个True，在非布尔型中，非0元素视为True
bools = np.array([False,False,True,False])
bools.any() # 有一个True嘛
bools.all() # 全是True嘛

# sort排序
arr.sort() # 在原数组上进行排序
np.sort(arr) # 不改变原数组

# 计算数组分位数
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]

# 加权平均
np.average(data, weights)
np.average([1,2,3], weights = [0.5,0.3,0.2])

# np.unique 找出数组的唯一值并返回已排序的结果

# 数组的集合运算，见下
```

![img](python%20notes.assets/7178691-80e85ae6b9c89ada.png)

## 线性代数

```python
# 矩阵乘法
x.dot(y) / np.dot(x,y) / x @ y
```

![img](python%20notes.assets/7178691-dcdb66e49e5f70ea.png)

## 伪随机数生成

numpy.random模块对Python内置的random进行了补充，增加了一些用于高效生成多种概率分布的样本值的函数。

```python
# 标准正态分布
samples = np.random.normal(size=(4, 4))

# random库中也有提供
import random
x = random.random() # 随机返回一个[0,1)区间内的数
x = random.randint(a,b) # 随机返回[a,b]区间内的一个整数
t = [1,2,3]
random.choice(t) # 随机从t中抽取一个数
# 其他部分np.random函数 见下图
```

![img](python%20notes.assets/7178691-97ba09c96dab93a2.png)

![img](python%20notes.assets/7178691-6ed04fae3d1178e2.png)

## 示例：随机漫步

我们通过模拟随机漫步来说明如何运用数组运算。先来看一个简单的随机漫步的例子：从0开始，步长1和－1出现的概率相等。

下面是一个通过内置的random模块以纯Python的方式实现1000步的随机漫步：

```python
# 内置random做法
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1 # 如果是1则返回1 否则为-1
    position += step
    walk.append(position)
import matplotlib.pyplot as plt
plt.plot(walk)

# numpy做法
nsteps = 1000
draw = np.random.randint(0,2,size=nsteps)
steps = np.where(draw > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)

(np.abs(walk) >= 10).argmax() # 首次抵达时间argmax返回第一个最大值位置

# 模拟多个随机漫步
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
plt.plot(walks.T)
```

# Pandas

```python
import numpy as np
import pandas as pd
```

## Getting data in/out

```python
# 批量读取文件夹下所有文件名
os.listdir(r'F:\balabala') # 读取F:\balabala下所有文件名称

# read in
df = pd.read_csv('abc.csv',header=None/0/1,index_col=['col1','col2'],skiprows=[0,1,2])

f = open('F:/YQY工作文件/20191022多因子筛选/3-data/ic02.csv')
df = pd.read_csv(f) # 若csv中有中文字体的解决方案

df = pd.read_hdf('*.h5','df')

df = pd.read_excel('*.xlsx',sheet_name = 'sheet2',index_col=None,na_values=['NA'],header=1) # 默认第一表单，选第二行为columns的名称，第一行就丢掉不要了

# write out
df.to_csv('*.csv',index=False) # 可以不打印索引
df.to_csv('*.csv',index=False,encoding='utf_8_sig') # 若出现中文乱码
df.to_hdf('*.h5','df')
df.to_excel('*.xlsx',sheet_name='sheet1')

# 输出到同一张excel里
writer = pd.ExcelWriter('temp.xlsx')

totaldf1.to_excel(writer,"5")
totaldf2.to_excel(writer,"10")
totaldf3.to_excel(writer,"15")
totaldf4.to_excel(writer,'20')
writer.save()

# 保留上一张表原来的内容
listdf = pd.read_excel('指数列表.xlsx').dropna()

temp = pd.DataFrame()
temp.to_excel("指数成分股.xlsx")
writer = pd.ExcelWriter("指数成分股.xlsx")

for i in range(len(listdf)):
    temp_code = listdf.loc[i, '指数代码']
    temp_date = listdf.loc[i, '指数成分日期']
    temp_name = listdf.loc[i, '指数名称']
    _,tempdf = w.wset("sectorconstituent","date={};windcode={}".format(
        temp_date, temp_code) ,usedf=True)
    
    book = load_workbook('指数成分股.xlsx')
    writer = pd.ExcelWriter("指数成分股.xlsx",engine='openpyxl')
    writer.book = book
    tempdf.to_excel(writer, temp_name + '_' + str(temp_date))
    writer.save()

```

![image-20200529100126172](python%20notes.assets/image-20200529100126172.png)

## Objection creation

```python
# Series
s = pd.Series([1,2,3,4,5,np.nan])

# dates
dates = pd.date_range(start,end,periods,freq,closed) # freq={'s','min','h','d','m','y','2s','3d',...} closed={None,'left','right'}

# dataframe
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

```

## Viewing data

```python
# summarise data
df.head() # default = 5
df.tail() # default = 5
df.index
df.columns
df.to_numpy()/df.asarray()/df.values # turn to numpy df.values不香了,df.values是老版本的，新版本使用前面两个
df.describe()
df.T

# 返回矩阵维度
df.shape[0] #返回行数
df.shape[1] #返回列数

# sort
df.sort_index(axis=1,ascending=False) # axis=1 sort_columns; axis=0 sort_rows; ascending=False 降序; ascending = True 升序
df.sort_values(by='A',ascending=True)
stockname/factorname = df[0].tolist() #用第一列的值对columns排序

df.rank(method = '', ascending = False, axis = 'columns')

def sort_priority(values, group): # 通过元组的排序原则，将values中的group内的值放在前面
    def helper(x):
        if x in group:
            return (0, x)
        return (1, x)
    values.sort(key=helper)
```

![image-20200529094140397](python%20notes.assets/image-20200529094140397.png)

## Selection

```python
# Getting -- []
df['col_name'] # select the specific column
df[0:3] # select rows
df['index_A':'index_B'] # select rows
	
# select by labels -- loc 利用标签的切片是左闭右闭的区间
df.loc['index_name'] # select the specific row by index_label
df.loc[:,'col_name']
df.loc[:,['col_name_A','col_name_C']]
df.loc['index_A':'index_C',['col_name_A','col_name_C']]
df.loc['index_A','col_B'] # select one value
df.at[dates[0],'col_B'] # ???

# select by position -- iloc
df.iloc[3] # select row 4 == df.iloc[3,:]
df.iloc[3:5,0:2]
df.iloc[[1,2,4],[0,2,3]] # select by position_list
df.iloc[1:3,:]
df.iloc[:,1:3]
df.iloc[1,1]
df.iat[1,1] # faster than prior method

# select by boolean -- logical index
df[df.A>0] # select rows by 'col_A > 0'
df[df>0] # if False, turn to NaN.
df2 = df.copy()
df2['E'] = ['one','one','two','two','three']
df4['E'] = ['one','two','one','three','two','four']
df4[df4['E'].isin(['two','four'])]
df4['F'] = [1,1,2,3,4,3]
temp = list([1,2])
df4[df4['F'].isin([1,2])]
df4[df4['F'].isin(temp)]

# isin()接受一个列表，判断该列中元素是否在列表中，并过滤
df['E'] = ['a','a','c','b']
df.E.isin(['a','c'])
df[df.E.isin(['a','c'])] # 过滤掉不在'a'、'c'中的

# setting
df4['G'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20190127',periods=6))
df.at[dates[0],'A'] = 0
df.iat[1,1] = 0
df.loc[:,'D'] = np.array([5]*len(df))
df.loc[dates[0]:dates[3],'E'] = 1
df[df<0] = -df # abs(df) <==> abs(df)

# reindex
df.index = range(len(df))

# 多重索引的切片
row.loc[['2015-03','2015-04'], [2.20,2.25]] # row是每一行
df.query('id' == 'a')
```



![image-20200528120251347](python%20notes.assets/image-20200528120251347.png)![image-20200827095953860](python%20notes.assets/image-20200827095953860.png)



```python
# 随机排列
sampler = np.random.permutation(5) # 随机产生一个0-4的排列
df.take(sampler) # 等价于 df.iloc[sampler,:]

# 随机抽样
df.sample(n=3, axis=0) #  随机抽3行
df.sample(n=3, replace=True) # 有放回，可重复
```

## Missing data

```python
# 检查是否有缺失值
pd.isnull()
pd.notnull()
df.isna()

# 数一下所有列不重复的有多少个
df.nunique(axis=0,dropna=True) 

# 数一数有多少个缺失值
series.value_counts()
pd.isnull(train_x).apply(lambda x: x.value_counts()) # 

# dropna()
df.dropna(how='any') # 一行中只要有1个nan就丢掉,how='all'全是nan丢掉
df.dropna(thresh = 100, axis='columns') # 对于每列而言，只有非nan数据大于100的，才会显示出来

# fillna()
df.fillna(value=5)
df.fillna(value={'A':0,'B':1,'C':2,'D':3,'E':4})
df.fillna(df.mean(axis=0)) # 列平均填充na
df.T.fillna(df.mean(axis=1)).T # 行平均填充na
df.fillna(method = 'bfill') # 后值填充
df.fillna(method = 'ffill') # 前值填充
df.fillna(method = 'ffill', limit=2) # 最多填充2个
rets[np.isnan(rets)] = 0
rets[np.isinf(rets)] = 0
# 【实例】用分组平均数填充缺失值
fill_mean = lambda x: x.fillna(x.mean())
data.groupby('key1').apply(fill_mean)
# 【实例】给定不同分组不同填充数
fill_values = {'group1': 1, 'group2': 2}
fill_func = lambda x: x.fillna(fill_values[x.name])
data.groupby('key1').apply(fill_func)

# replace
df.replace(-1000, np.nan)
df.replace(np.nan, 0)
df.replace({-999:np.nan, -1000:0})
df.replace([-999,-1000],[np.nan,0])

# 把缺失值/异常值所在的位置拿出来
df[df.isnan().any(1)]
df[df>3.any(1)]
data[np.abs(data) > 3] = np.sign(data) * 3 # np.sign根据符号返回1，-1

# 用B序列来填充A序列的缺失值 a可以是series，也可以是dataframe
a.combine_first(b) # 想象成把b放在a的上面，然后从下面俯视 覆盖上去
```

## Operations

```python
# stats
df.mean() # mean by column
df.mean(1) # mean by row
s.shift(1) # 向前平移一天 s是1.2的数据，s.shift(1)是1.1的数据
df.sub(s,axis='index') # axis=0/axis='index', df=df-s by columns

# 返回最大最小值索引
df.idxmax()
df.idxmin()
```

![image-20200428101108619](python%20notes.assets/image-20200428101108619.png)

```python
# apply
df.apply() # default axis=0,按行执行; axis=1,按列执行
df.apply(np.cumsum) <==> df.cumsum()# 累加
df.apply(lambda x: x.max() - x.min(),axis='columns') # x代表每行

# map
series.map() # 对每个元素进行映射，可以用函数或者dict进行映射

# applymap 元素级应用
df.apply() # 是对每列
series.map() # 是对每列的每个元素
df.applymap() # 是对每个元素

# Histogramming 直方图
s = pd.Series(np.random.randint(0,7,size=10))
s.value_counts()

# 插入列
df.insert(3,"score",0) #在第4列插入"score"列,值为0
df['score'] = 0 # 在最后插入新的列

# 插入行
df.loc['aaa'] = list

# 删除行/列
df.drop(['index_name1','index_name2'],inplace=True)
df.drop(df.index[0:3],inplace=True)
df.drop(df.index[[0,3]],inplace=True)
df.drop(df.index[3],inplace=True)
df.drop(['A'],axis=1) # axis=1删除列
df.drop_duplicates('col_name',keep  = 'first/last',inplace=True) # 删除重项
df.duplicated() # 判读是否有重复项
del df['index_name1']	

# 去除重复项
pd.unique()
```

```python
# 算术运算
df1.add(df2, fill_value = 0)

df - 1 # 每个位置都减1
df - df['A'] # 每行减去
```

![image-20200529091034225](python%20notes.assets/image-20200529091034225.png)

```python
# str方法
df.str.lower() # 全部改成小写
df.str.replace('abc','') # 全字符串查找并替代
df.str.any_method_in_str # 所有字符串方法都可以使用
df.str.contains()
```

![img](python%20notes.assets/7178691-a634364ed6d5d5c5.png)

## Index

```python
# 重命名行列
df.index = [list]
df.set_index('column_name',inplace=True)

df.index = df.index.map()
df.rename(index={},columns={}) # 这个方法很不错，可以只替换一个
df.columns.map(lambda x:'df' if x=='dominant_future' else x) # 将dominant_future改为df

# 将index转换成一列
df.reset_index() # 将原来的index放到数据中，创建新的index

# reindex 将索引按照新索引的顺序排列，不存在则引入缺失值，直接补全自然日并用前值填充，reindex也可以用在index上
df.reindex(['a','b','c'])
df.reindex(index=['a','b'])
df = get_price('000001.XSHG',start_date = '2019-01-01',end_date = '2019-01-31',fields=['close'])
df.reindex(pd.date_range('20190101','20190131'),method='ffill')
df.reindex(columns=list) # 按照新的列排序

# column重新排列
pd.DataFrame(data, columns = ['B','A','C'])
df.sort_index(axis=0, ascending = False) # ascending是升序的意思
```

![img](python%20notes.assets/7178691-efa3dbd4b83c61ec.jpg)

![img](python%20notes.assets/7178691-5499d14f0e2cd639.jpg)

## Merge

```python
# concat
pd.concat([df1,df2],axis=0) # default=0 上下按行拼，axis=1左右拼

# join/merge # merge可以取并集或者交集 %in%
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False)  
	
# append
df.append(s,ignore_index=True) # 只可附在最后一行
```

## Grouping

```python
# 按照类型分组
df.groupby(df.dtypes) # 可以用类型分组
df.groupby('key1') # 根据columns分组
df.groupby(mapping) # 根据映射分组，可以是dict、series、函数

# groupby函数的参数
df.groupby('key1',as_index=False).max() # index用新生成的arange代替
df.groupby('key1',group_keys=False).apply() # index返回数据原来的index

# groupby迭代
for name,group in df.groupby('key1'): # 产生一个分组名，一个分组df
    print(name)
    print(group)
pieces = dict(list(df.groupby('key1'))) # 设置字典，将分组名和分组数据相匹配

# groupby聚合运算
df.groupby(['key1','key2']).size() # 计算分组大小
grouped['data1'].quantile(0.9) # 计算分位数

# 自定义聚合函数
def alpha1(df):
    return df.high.max() - df.low.min()
alpha1(df.groupby('stock'))

def peak_to_peak(arr):
    return arr.max() - arr.min()
df.groupby('key1').agg(peak_to_peak)

df.groupby('key1').agg([peak_to_peak,'std','mean']) # 进行多个聚合运算
df.groupby('key1').agg([('average','mean'),('volatility','std')]) # 比上一行多了列名
df.groupby('key1').agg({'data1':['mean','max','min'],'data2':'sum'}) # 对data1做三个聚合运算，对data2做求和运算

df.groupby('key1').apply(func)
# 【实例】用分组平均数填充缺失值
fill_mean = lambda x: x.fillna(x.mean())
data.groupby('key1').apply(fill_mean)
# 【实例】给定不同分组不同填充数
fill_values = {'group1': 1, 'group2': 2}
fill_func = lambda x: x.fillna(fill_values[x.name])
data.groupby('key1').apply(fill_func)

# 设置多重指标
columns = pd.MultiIndex.from_arrays([['行情','行情','行情','量','量','幅度'],
          ['OPEN','HIGH','LOW','VOLUME','AMT','PCT_CHG']],names=['s1','s2'])
hier_df = pd.DataFrame(df_wss.values, columns=columns,index=df_wss.index)

# 取索引
df.index.get_level_values(0).unique() # 取1级索引

# 面元划分、离散化、分组 pd.cut长度相等（每个bucket长度一样长） / pd.qcut大小相等（每个bucket里的数量一样多）
pd.cut(raw_series, bin, right=False, labels=group_names) # right设置左右开闭
pd.cut(raw_series, 4, right=False, labels=group_names, precision=2) # 均匀分成四组，限定2位小数
# pd.qcut 可以将每个面元内的元素数量划分的一样
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
pd.value_counts(cats)

## 分位数分组 + 聚合运算
quartiles = pd.cut(frame.data1, 4)
frame.data1.groupby(quartiles).describe()

def get_stats(group):
    return {'min':group.min(), 'max':group.max(),
            'count':group.count(), 'mean':group.mean()}
grouped = frame.data1.groupby(quartiles)
temp = grouped.apply(get_stats).unstack()

# 分层抽样
# 【实例】扑克牌抽牌，每个花色抽两张
suits = ['H','S','C','D']
card_val = (list(range(1,11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2,11)) + ['J','Q','K']
card = []
for suit in suits:
    card.extend(str(num) + suit for num in base_names)
card
deck = pd.Series(card_val, index = card)
def draw(deck, n=5):
    return deck.sample(n)
deck.groupby([temp[-1] for temp in card]).apply(draw,n=2) # 根据花色分组，每个花色抽2张

```

![image-20200607102923212](python%20notes.assets/image-20200607102923212.png)

## Reshaping

```python
# 生成multiindex
tuples = list(zip(*[['s1','s1','s2','s2'],['d1','d2','d1','d2']]))
index = pd.MultiIndex.from_tuples(tuples, names=['stock', 'date'])

# stack/unstack多维数组的展开与折叠
df.stack() # 将多列压缩进一列，row为一级指标，col为二级指标
df.unstack() # -1将最后一级指标展开,0将一级指标展开,1将二级指标展开

# pivot_table 指定维度
df.pivot_table(values='',index='',columns='') # 指定x,y

# melt函数，宽表格变长表格
melted = pd.melt(df, ['key']) # 将宽表格按照key分组变成长表格
reshaped = melted.pivot('key', 'variable', 'value') # 再转化回去 index=key, columns=var

# 层次化索引 多重索引
data = pd.Series(np.random.randn(9),index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],[1, 2, 3, 1, 3, 1, 2, 2, 3]])
data = pd.read_csv('data.csv',index_col = [0,1])

# 互换索引等级
frame.swaplevel('key1', 'key2')

# 获得多重索引值
df.index.get_level_values(level=1)
```

### 透视表和交叉表

```python
# pivot table
df = pd.read_excel('20200423数据总表.xlsx') # 豆粕期货数据总表
df.pivot_table(index = ['strike_price']) # 默认计算分组平均数
df.pivot_table(['ambiguity','RDVar'], index = ['strike_price']) # 只聚合模糊性和方差
df.pivot_table(['ambiguity','RDVar'],index=['strike_price'],columns=['pos_or_neg']) # 聚合的时候根据pos_or_neg再分组计算平均数
df.pivot_table(['ambiguity','RDVar'],index=['strike_price'],columns=['pos_or_neg'],margins=True) # 同时保留pos_or_neg = [0,1,all] 
df.pivot_table(....,aggfunc = len/'mean', fill_value = 0) # 填充缺失值、其他聚合方法

# crosstab 计算分组频率/频次
pd.crosstab(df.strike_price, df.pos_or_neg, margins = True)
```

![image-20200607201337852](python%20notes.assets/image-20200607201337852.png)

## Time series

### Datetime操作

```python
# reference:
## https://www.cnblogs.com/nxf-rabbit75/p/10662025.html

from datetime import datetime
dates = pd.date_range(start='2019-04-01', periods=20, freq='B', normalize=True) # normalize表示标准化，全部换到午夜00:00
ts = pd.Series(np.random.randn(20),index=dates)
ts[::2] # 从第一个开始取，取1,3,5,7,9...
stamp = ts.index[0] # stamp.dtype = timestamp
stamp = ts.index[2:5]
ts[stamp] # 取到stamp的值
ts['20190403'] # 可解读为日期的字符串也可以读取
ts['20190403':]
ts[datetime(2019,4,13):]
ts['20190403':'20190407']

longer_ts = pd.Series(np.random.randn(200),index=pd.date_range('4/1/2019',periods=200))
longer_ts.loc['2019-4'] # 取四月份的数据 also .loc['2019/04']
longer_ts.loc['2019'] # 取2019年数据

ts.truncate(after='4/3/2019')

# 锚点偏移
from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()
now + MonthEnd()
now + MonthEnd(2)
offset = MonthEnd() 
offset.rollforward(now) # 向后滚动到最近的月底
offset.rollback(now) # 向前滚动到最近的月底
ts.groupby(offset.rollforward).mean() # 先把这些日期都放到月底，分组平均 
ts.resample('M').mean() # 等价于上行
datetime.today() - pd.tseries.offsets.Week(weekday=0) # 上周一，如果不指定参数就是周日
```

![img](python%20notes.assets/7178691-c8614ddbd10793ca.png)

![img](python%20notes.assets/7178691-8da46ba96544b071.png)

![img](python%20notes.assets/7178691-3ca410609195edc4.png)

```python
# 差分
df.diff()
```

### 时区操作 timezone

```python
# 父class
import pytz
pytz.common_timezones # 看看有哪些常见时区
pytz.timezone('America/New_York') # 看看纽约时区的信息

# 默认情况下 pandas用None时区
print(ts.index.tz) # none

# 用时区集生成时区
pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC') # UTC是国际标准

#从None到本地化的转换是通过tz_localize方法处理的
ts_utc = ts.tz_localize('UTC')

# tz_convert 改变时区
ts_utc = ts.tz_localize('utc') # 设定当地时间为标准时/*
ts_utc.tz_convert('US/Eastern') # 进行时区转化
stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow

# 不同时区的时间戳加在一起会全部变成UTC时区，即标准时区
```

### 时期操作 Period

```python
# !!!核心是通过period构造时期，再用asfreq选择每个时期的第几天第几个小时

p = pd.Period(2007, freq='A-DEC') # 2017年12月最后一个日历日
p + 5 # 2012年
p - 2 # 2005年

# 创建规则的时期范围
rng = pd.period_range('2000-01-01', '2000-06-30', freq='M') # ['2020-01','2020-02'...]

# 字符串数组也可以创建时期index  PeriodIndex
values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')

# 时期的频率转换
p = pd.Period(2007, freq='A-DEC') # 2017年12月最后一个日历日
p.asfreq('M', how='start') # 2007-01
p.asfreq('M', how='end') # 2007-12
'''如果不以A-DEC作为年度划分依据，start和end对应的就会不一样了，想象成一个滚动的窗口'''
# 对时期进行操作
ps.asfreq('m','e'+1) # e是end，s是start，对periodindex操作 +1就是加1个频率单位 +1月	
p = pd.Period('2012Q4', freq='Q-JAN') # 以1月为每年的截止
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60 # 每季度倒数第二个交易日下午四点
p4pm.to_timestamp()

# 转化成timestamp 生成新的index
rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = pd.Series(np.arange(len(rng)), index=rng)
new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()

# 用数组生成periodindex
index = pd.PeriodIndex(year=data.year, quarter=data.quarter,freq='Q-DEC')
```

### 类型转换 Time conversions

![img](python%20notes.assets/1252882-20190619203503264-949480978.png)

```python
# 其他类型转datetime类型
pd.to_datetime(df)

# 其他类型转period类型
ts.to_period() # 将datetimeindex转换为periodindex

# to_timestamp 将periodindex投影到datetimeindex
ts.to_timestamp(how='end')

ts.to_period('s/min/h/d/m/y') 
ps = ts.to_period()
ps.to_timestamp()
```

### 重采样 resample

```python
# resample 改变时间戳
rng = pd.date_range('20120101',period=100,freq='20s')
ts = pd.Series(np.random.randint(0,500,len(rng)),index=rng)
ts.resample('5min').sum() # 按照5分钟分组，求和
ts.resample('5min',closed='right').sum() # right (9:00,9:05]  left [9:00,9:05)
ts.resample('5min', closed='right',label='right', loffset='-1s').sum() # 校正1秒以示范围
# 也可以通过shift来校正
ts.resample('5min', closed='right',label='right').sum().shift(-1,freq='s')

# OHLC open、high、low、close
ts.resample('5min').ohlc() # 返回开盘价、收盘价、最高价、最低价

# 升频采样 从低频到高频，缺失值用NaN补齐
df_daily = frame.resample('D').asfreq() # 不加asfreq等于没执行操作 加了asfreq意为执行换频率
frame.resample('D').ffill(limit=2) # 用ffill也行

# 对于period进行重采样
annual_frame.resample('Q-DEC', convention='end').ffill() # 会麻烦一些，因为要指定哪端放置
```

![image-20200608143134870](python%20notes.assets/image-20200608143134870.png)

### 移动窗口与拓展窗口

```python
# 移动平均
housedf.rolling(3,min_periods=1).mean() # 对列做移动平均
housedf.rolling('20D').mean() # 对不规则的时间序列进行移动平均，不一定每个window下都是20天

# 拓展平均
expanding_mean = appl_std250.expanding().mean() # window长度从1到最大

# 加权移动平均 EWMA 根据距离今日的时间长短，赋予不同的权重
ewma60 = aapl_px.ewm(span=30).mean()

# 移动相关系数
df.rolling(window=60,min_period=10).corr(spx_rets)
df.ambiguity.rolling(60,min_periods=10).corr(df.close_ret).plot() # 模糊性和收益率
```

## Categoricals

```python
df['grade'] = df['row_grade'].astype('category')
df['grade'].cat.categories = ['very good','good','bad']
df['grade'] = df['grade'].cat.set_categories(['very bad','bad',
  'medium','good','very good']) # 设置好所有的分类
df.sort_values(by='grade',ascending=False) # 按照分类排序
df.groupby('grade').size() # 显示每个分类的规模
```

## Plotting

```python
# 是一个简化版的matplotlib 高度集合
s.plot() # series
df.plot() # dataframe
```

![image-20200606135938548](python%20notes.assets/image-20200606135938548.png)

![image-20200606141514247](python%20notes.assets/image-20200606141514247.png)

```python
df.plot.bar(stacked = True) # 竖着的，堆叠着的
df.plot.barh() # 横着的
s.value_counts().plot.bar()

df.plot.hist(bins=50) # 直方图
df.plot.density() # 概率密度图

```

## one-hot encoding

```python
pd.get_dummies()
```

## style

```python
# 给dataframe上颜色
def color_t_red(val):
    color = 'red' if abs(val) > 1.96 else 'white'
    return 'color:%s'% color
s = temp_t.style.applymap(color_t_red)
```

![image-20200821133520324](python%20notes.assets/image-20200821133520324.png)

# Visualization

## Matplotlib

### Elements

- figure: 整张图表，最大的一个画布叫做figure，包含(title, axes)
- axes: 子图表，包含(titles, figure legends, etc)，set_title(), set_xlabel(), set_ylabel()
- axis: 是axes的坐标轴，往往一个axes有两个axis (2D)，但也可以有三个axis, axis限定数据范围(当然也可以在axes里设定via set_xlim(), set_ylim())，属性有ticks, ticklabels, locator(决定ticks的位置), formatter(决定ticklabel的字符串格式)
- Artist: 艺术对象的总称，所有的你可以看到的东西都是artist(Figure, Axes, Axis, Text, Line2D, collection, Patch ...)，当图像被渲染之后，所有的artists被画在画布(canvas)上

![image-20191224134346982](python%20notes.assets/image-20191224134346982.png)

### Plotting functions

While plotting, you'd better input in type of **np.array** instead of **pandas** or **np.matrix**.

```python
df = df.values
np_matrix = np.asarray(np_matrix)
```

For functions in the pyplot module, there is always a "current" figure and axes.

```python
x = np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title('Simple Plot')

plt.legend()

plt.show()
```

Recommended function signature:

```python
def my_plotter(ax, data1, data2, param_dict):
    '''
    A helper function to make a graph
    
    Parameters
    ----------
    ax: Axes
    	The axes to draw to
    	
    data1: array
    	The x data
    
    data2: array
    	The y data
    
    param_dict: dict
    	Dictionary of kwargs to pass to ax.plot
    	
    Returns
    -------
    out: list
    	list of artists added
    '''
    out = ax.plot(data1, data2, **param_dict)
    return out
```

Use like this:

```python
# subplots = 1
data1, data2, data3, data4 = np.random.randn(4,100)
fig, ax = plt.subplots(1,1)
my_plotter(ax, data1, data2, {'marker':'x'})

# subplots = 2
fig, (ax1,ax2) = plt.subplots(1,2)
my_plotter(ax1, data1, data2, {'marker':'x'})
my_plotter(ax2, data3, data4, {'marker':'o'})
```

### Performance

Rendering performance can be controlled by the two parameters: **path.simplify** and **path.simplify_threshold**.

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Setup, and create the data to plot
y = np.random.rand(100000)
y[50000:] *= 2
y[np.logspace(1, np.log10(50000),400).astype(int)] = -1
mpl.rcParams['path.simplify'] = True # whether to simplify the lines

mpl.rcParams['path.simplify_threshold'] = 0.0 # default = 1/9
plt.plot(y)
plt.show()

mpl.rcParams['path.simplify_threshold'] = 1.0
plt.plot(y)
plt.show()
```

Another way to rendering performance is as follows: 

```python
import matplotlib.style as mplstyle
mplstyle.use('fast')
mplstyle.use(['dark_background', 'ggplot', 'fast'])
```

```python
import matplotlib.pyplot as plt

# 显示中文字体，负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 负号乱码

# create a new figure
fig = plt.figure() # an empty figure with no axes
fig.suptitle('No axes on this figure') # add a title
fig, ax_lst = plt.subplots(2,2) # a figure with 2*2 grid of axes

# plot
x = np.linspace(0,2,100)
plt.plot(x, x, label='linear')     
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label') 
plt.ylabel('y label')
plt.suptitle('Simple Plot',fontsize=20) # title    
plt.legend() # show the legend
plt.show()   

# 主次坐标轴
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
 
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1,color=g')
ax2.set_ylabel('Y2,color=b')
 
plt.show()

# 画矩形代码
import matplotlib.transforms as transforms
trans=transforms.blended_transform_factory(ax1.transData,ax1.transAxes)
ax1.fill_between([df.index[1],df.index[20]],0,1,transform=trans,alpha=0.1,color='r') # 0从底部起 1到顶部止


# 画宏观周期实例
def draw_single_cycle(df,n,benchmark=False):
    # 画布与格式初始化
    # df[df.columns[0]] = round(df[df.columns[0]].rolling(3,min_periods=1).mean())
    large = 32; med = 16
    params = {'axes.titlesize' : large,
              'legend.fontsize' : med,
              'figure.figsize' : (16,10),
              'axes.labelsize' : med,
              'xtick.labelsize' : med,
              'ytick.labelsize' : med,
              'figure.titlesize' : large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style('white')
j
    # 画布大小
    fig = plt.figure(1)
    ax1 = plt.subplot(211)
    ax2 = ax1.twinx()
     ax1.plot(df[df.columns[1]],'lightcoral',alpha=1,linewidth=3,label=df.columns[1])  	ax2.plot(df[df.columns[2]],'lightsteelblue',alpha=1,linewidth=3,label=df.columns[2])	trans=transforms.blended_transform_factory(ax1.transData,ax1.transAxes)
    for i in range(len(df)-1):
        rectrange = [df.index[i-1],df.index[i]]
        if df.iloc[i][0] == 2:
            ax1.fill_between(rectrange,0,1,linewidth=0,transform=trans,alpha=0.1,color='r')    
        elif df.iloc[i][0] == 0:
            ax1.fill_between(rectrange,0,1,linewidth=0,transform=trans,alpha=0.1,color='g')
        elif df.iloc[i][0] == 1:
            ax1.fill_between(rectrange,0,1,linewidth=0,transform=trans,alpha=0.1,color='b')
        else:
            pass
    # 标题
    title = ' {}-{}'.format(df.index[0].strftime('%Y.%m.%d'),df.index[-1].strftime('%Y.%m.%d'))
    temptitle = 'Modified {} cycle '.format(df.columns[0]) + 'MA' + str(n)
    plt.suptitle(temptitle + title)
    # 图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.legend(handles1+handles2, labels1+labels2, loc='upper right')
    
    ax3 = plt.subplot(212)
    ax3.axis('off')
    results = cycledescribe(df)
    ax3.text(0.5, 0.5,results, horizontalalignment='center', verticalalignment='center',fontsize=24)
    
    plt.show()
    os.chdir(path2)
    fig.savefig("{}.png".format(temptitle))
```

![preview](python%20notes.assets/v2-e395bc08e394a340137909f4bae42784_r.jpg)

![image-20210611171610011](python%20notes.assets/image-20210611171610011.png)

![preview](python%20notes.assets/v2-9ef6b6270a908e1678ac52304acf1b97_r.jpg)

![preview](python%20notes.assets/v2-37a250d3dd0fda25c21caca7b05b4c3e_r.jpg)

### 画布与子图

```python
import matplotlib.pyplot as plt
%matplotlib notebook

# 画布创建与子图设置
## 第一种画布与子图设置方式
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax1.plot()
ax2.plot()

    ## 第二种画布与子图设置方式
    fig, ax = plt.subplots(2,3,sharex=True,sharey=True,figsize=(12,8))
    ax[0,0].plot()
    ax[0,1].plot()

# 子图间距
subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None) # wspace=1 表明宽间距是子图宽度的1倍
```

![image-20200606111618154](python%20notes.assets/image-20200606111618154.png)

### 颜色、标记和线型

```python
ax.plot(x, y, 'go--')
ax.plot(x, y, linestyle = '--', color = 'g', marker = 'o', drawstyle = 'steps-post', 	label = 'balabala')
# plt.legend() / ax.legend(loc = 'best')
ax,scatter(x, y, s=16) # s是点的size	

color = ['#004c94', '#f05f5f', '#65a2e5', '#ffbdc8'] # 兴业的配色
```

### 设置标题、轴标签、刻度以及刻度标签

```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())

# 设置刻度和刻度标签
ticks = ax.set_xticks([0, 250, 500, 750, 1000]) # 设置位置
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
         rotation=30, fontsize='small') # 设置标签 可以不要

# 设置标题和坐标轴名称
ax.set_title('demo')
ax.set_xlabel('Stages')

props = {
    'title': 'demo',
    'xlabel': 'Stages'
}
ax.set(**props)
```

### 注解以及在子图上绘图

```python
# ax.text
ax.text(x, y, 'Hello world!',
        family='monospace', fontsize=10)
```

```python
# 标注次贷危机期间的重要事件
from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('examples/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')
```

![image-20200606120139446](python%20notes.assets/image-20200606120139446.png)

```python
# 形状
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3) # 矩形
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3) # 圆形
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], # 多边形
                   color='g', alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
```

### 图像存储

```python
plt.savefig('figpath.png', dpi=400, bbox_inches='tight')
# dpi是分辨率，bbox_inches是周围白边的大小 tight默认剪除空白
```

### 参数全局配置

```python
plt.rc('figure', figsize=(10,10)) # figure,axes,xtick,ytick,grid,legend

font_options = {'family':'monospace',
               'weight':'bold',
               'size':'small'}
plt.rc('font', **font_options) # **应该是一个解包的过程

# 这个字体很好看，times new roman
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal', # or italic斜体
        'font.weight':'normal', #or 'blod'
        'font.size': 16,#or large,small
        }
plt.rcParams.update(params)

# 画图模板
## 图例
plt.legend(handles1+handles2, labels1+labels2, framealpha=0)
```

### 气泡图

```python
crowd = cal_ind_crowd(code_list, name_list, end_date=yesterday)
fig,ax = plt.subplots(figsize=(20,16))
ax.scatter(totaldf['pb_lf'],totaldf['pe_ttm'], s=np.array(crowd)*300,
c=np.array(crowd),linewidth=2.5,marker='o', cmap='Reds',alpha=0.7)
plt.xlabel('PB quantile',fontsize=28)
plt.ylabel('PE quantile',fontsize=28)
for i in totaldf.index:
plt.annotate(i, xy = (totaldf.loc[i,'pb_lf']+0.002, totaldf.loc[i,'pe_ttm']),fontsize=20)
#plt.colorbar(label = '拥挤度')
plt.show()
#os.chdir(r'C:\Users\YANGQINGYUAN\Desktop')
fig.savefig('temp.png', dpi=400, bbox_inches='tight')
```

![temp](python%20notes.assets/temp.png)

### xticks间隔修改

```python
northdf.index = northdf.index.astype(str)
fig,ax = plt.subplots()
ax.bar(northdf.index, northdf['北向资金净流入（亿元）'])
xticks=list(range(0,len(northdf.index),10)) # 这里设置的是x轴点的位置（40设置的就是间隔了）
xlabels=[northdf.index[x] for x in xticks] #这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
xticks.append(len(northdf.index))
xlabels.append(northdf.index[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=40) # 旋转40度
ax.set_title('北向资金净流入（亿元）')
plt.show()

```

![image-20210709194436660](python%20notes.assets/image-20210709194436660.png)

### 报错修复 debug

**UserWarning: agg**

```python
import matplotlib; matplotlib.use('TkAgg')
```

![image-20200814112354367](python%20notes.assets/image-20200814112354367.png)

### 案例

#### 兴业模板

```python
barcolor = ['#004c94', '#f05f5f', '#65a2e5', '#ffbdc8', '#bfbfbf', '#ffc000', '#e02622']
linecolor = ['#e02622', '#08287f', '#bfbfbf', '#ff9933', '#5acdf5', '#f05f5f', '#737373']
barlinecolor = ['#e02622', '#bfbfbf', '#5acdf5', '#ff9933', '#ffbdc8', '#f05f5f', '#08287f']

params = {'axes.titlesize' : 36,
          'legend.fontsize' : 20,
          'figure.figsize' : (16,10),
          'axes.labelsize' : 20,
          'xtick.labelsize' : 20,
          'ytick.labelsize' : 20,
          "font.family":'serif',
          "mathtext.fontset":'stix',
          "font.serif": ['SimSun'],
          'figure.titlesize' : 36,
         'axes.unicode_minus': False}
plt.rcParams.update(params)
```



## seaborn

```python
import seaborn as sns

# 散点图 + OLS估计出的总体模型
sns.regplot('ambiguity', 'RDVar', data=df)

# 散点图 + 概率密度图
sns.pairplot(df.iloc[:,0:4], diag_kind='kde', plot_kws={'alpha': 0.2})
```

# Time

```python
# Timer
totalstart = time.clock()
totalend = time.clock()
print('Running success in {:.2f}s.'.format(totalend-totalstart))
```

# Datetime

```python
import datetime
from datetime import datetime
now = datetime.now()
now.year, now.month, now.day
delta = datetime(2011,1,2) - datetime(2001,1,2)
delta.days

from datetime import timedelta
now + 2 * timedelta(days=10)

# 字符串和日期的相互转化
tm.strftime('%I%M%S %p') # 04:01:30 PM
    ''' 格式化字符
    %a星期的简写。如 星期三为Web
    %A星期的全写。如 星期三为Wednesday
    %b月份的简写。如4月份为Apr
    %B月份的全写。如4月份为April
    %c: 日期时间的字符串表示。（如： 04/07/10 10:43:39）
    %d: 日在这个月中的天数（是这个月的第几天）
    %f: 微秒（范围[0,999999]）
    %H: 小时（24小时制，[0, 23]）
    %I: 小时（12小时制，[0, 11]）
    %j: 日在年中的天数 [001,366]（是当年的第几天）
    %m: 月份（[01,12]）
    %M: 分钟（[00,59]）
    %p: AM或者PM
    %S: 秒（范围为[00,61]，为什么不是[00, 59]，参考python手册~_~）
    %U: 周在当年的周数当年的第几周），星期天作为周的第一天
    %w: 今天在这周的天数，范围为[0, 6]，6表示星期天
    %W: 周在当年的周数（是当年的第几周），星期一作为周的第一天
    %x: 日期字符串（如：04/07/10）
    %X: 时间字符串（如：10:43:39）
    %y: 2个数字表示的年份
    %Y: 4个数字表示的年份
    %z: 与utc时间的间隔 （如果是本地时间，返回空字符串）
    %Z: 时区名称（如果是本地时间，返回空字符串）
    %%: %% => %
    '''
datetime.strptime('2011-01-03', '%Y-%m-%d') # str转datetime

datestrs = ['7/6/2011', '8/6/2011']  
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs] # str列表转datetime列表

from dateutil.parser import parse # 直接解析str，转datetime
parse('2011-01-03')
parse('3/1/2011', dayfirst = True)

datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
pd.to_datetime(datestrs) # str列表转datetime
idx = pd.to_datetime(datestrs + [None]) # 保留一位NaT not a time

# several immutable classes
'''
datetime.date:表示日期的类,year,month,day;
datetime.time:表示时间的类,hour,minute,second,microsecond;
datetime.datetime:表示日期时间;
datetime.timedelta:表示时间间隔;
datetime.tzinfo:时区相关;
'''
# 其他type转换为datetime type
df.index = pd.to_datetime(df.index)

# class date
## class datetime.date(year, month, day)
day = datetime.date(2019,10,10)
day.max
day.min
day.resolution
day.today()
day.fromtimestamp(timestamp) # 给定时间戳返回date对象
day.weekday() # 定义一个datetime.date类，today.weekday()
day.isoweekday() # 1-7 区别于上面的0-6
day.isoformat() # 按照XXXX-XX-XX的标准格式输出
day.strftime('%y%m%d') # 按给定格式输出 %Y=2019 %y=19
today + day.resolution # 加一天
today - day # return 时间间隔 timedelta()
today < day # False / True

# class time
## class datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None) 只能比较 不能加减
tm = datetime.time(16,1,30)
tm.replace(hour=14)
tm.isoformat() # 16:01:30

# class datetime
## class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
dtm = datetime.datetime.today()
dtm.date()
dtm.time()
dtm.replace()
dtm.weekday()
dtm.isocalendar() # 迷
dtm.isoformat()
dtm.ctime() # 返回日期的字符串
## 可以加减 可以比较

# class timedelta
## class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

# relativedelta

```python
from dateutil.relativedelta import relativedelta

# 锚点偏移 一年前、一个月前等
```

![image-20210824215631314](python%20notes.assets/image-20210824215631314.png)

# Modelling

## Scipy

![image-20200428102406939](python%20notes.assets/image-20200428102406939.png)

```python
# 计算概率密度
from scipy import stats
stats.norm(mu, sd).pdf(x)
```

## statsmodels

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
mod = sm.OLS(y, X) 
res = mod.fit()
print(res.summary())

res.params # 参数
res.rsquared # R方

sm.stats.linear_rainbow(res) # 线性检验 不懂是啥 返回F值和p值

y, X = dmatrices('y ~ x0 + x1 + 0', data = df) # 无截距回归

y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data = df) # 在其中加入公式
y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data) # 直接标准化和中心化
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data) # 将不同列相加
y, X = patsy.dmatrices('y ~ C(x0)', data) # C函数 将数据列变成分类列

results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.predict(data[:5])

# HAC调整
mod = sm.OLS(y, X) 
res = mod.fit(cov_type='HAC',cov_kwds={'maxlags':8})  # 选样本数^0.25
res.summary()
```

![image-20211220152237899](python%20notes.assets/image-20211220152237899.png)

# Spider

## scrapy

### Introduction

**特性：**采用**twisted异步**网络框架，加快下载速度。

​	异步：调用在发出之后，这个调用就直接返回了，不管有无结果，直接开始下一步。

**scrapy框架**

![image-20210414154318603](python%20notes.assets/image-20210414154318603.png)

- scheduler：调度器，存放url队列。
- downloader：下载器，发送请求。
- spiders：提取url，放到scheduler里；提取数据，发送给item pipeline。可以创建多个spiders。
- pipeline：数据队列，存储数据。可以有多个，不同的pipeline处理不同的spider。
- scrapy engine：作为中间串联，使得上面四个模块分割开，这样容错率高一些。
- middlewares：中间键，可以处理request、response，不过spider middlewares一般不处理items，因为items放在pipeline处理。

![image-20210415100719493](python%20notes.assets/image-20210415100719493.png)

### framework

```python
# 创建一个爬虫项目
scrapy startproject MySpider # MySpider是项目名称

# 生成一个爬虫
cd MySpider
scrapy genspider itcast itcast.cn # 爬虫名叫做itcast，itcast.cn是允许爬取的范围
'''这样就可以在MySpider/spider目录下生成一个itcast.py，里面的参数start_urls就是最开始请求的url地址'''

## 操作1：编写itcast.py，找链接，定义url的爬取方式
'''查看一下要爬取的url链接里是否有你要的数据和信息，检查+查看源代码，如果有，那就是这个链接了，#后面的不要，把这个链接放入到start_urls'''
'''parse函数处理start_urls中的response'''

'''需要查看robots.txt协议，如果禁止，则要在settings里面修改robottxt_obey = False'''
ROBOTSTXT_OBEY = False

'''如果修改了之后，body还是很少，说明对方把你给禁止了，则要在settings里修改USER_AGENT'''
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36' # 网页-检查-network-headers-user_agent

def parse(self, response):
    ret1 = response.xpath("//div[@class='maincon']//h2/text()").extract() # extract()是只提取文字，这里爬取的列表不是一个普通的列表，普通的列表是没有extract方法的。
    # print(ret1)
    yield ret1 # 减小内存占用

def parse(self, response):
    # 分组爬取，每组提取第一个字符串
    li_list = response.xpath("//div[@class='maincon']//li")
    for li in li_list:
        item = {}
        item['name'] = li.xpath(".//h2/text()").extract_first() # extract_first()要优于extract()[0]，因为如果是后者，爬取了一个空列表的话，[0]的切片就会报错，而extract_first()方法返回none，不会报错
        item['title1'] = li.xpath(".//h3//span/text()").extract_first()
        item['title2'] = li.xpath(".//h3[2]//span/text()").extract_first() # 这里的[2]表示第二个h3。
        # print(item)	
    	yield item # 将item上传到pipeline里去，列表是不被允许返回的，只允许request、baseitem、dict、none

## 操作2：编写pipeline，处理和存储数据
'''在爬虫程序里parse函数yield结果之后，数据会被传输到pipeline'''
'''先要到settings里面开启pipeline，把ITEM_PIPELINE的注释取消掉'''
ITEM_PIPELINES = {'MySpider.pipelines.MyspiderPipeline': 100,
                  'MySpider.pipelines.MyspiderPipeline2': 300} # 300是距离，可以定义多个pipeline，优先执行距离近的pipeline
'''再到pipeline中，编写数据处理的代码'''
class MyspiderPipeline:
    def process_item(self, item, spider):
        item['hello'] = 'world' # 编写操作代码
        return item # 返回值
    
class MyspiderPipeline2:
    def process_item(self, item, spider):
        print(item)
        return item
    
## 操作3：启动爬虫
'''进入MySpider目录下启动爬虫，这个目录下还有scrapy.cfg文件。 '''
scrapy crawl itcast 
LOG_LEVEL = 'WARNING' # 在settings里面设置，log分为debug、info、warning、error四个等级，这里只显示前两级
```

### **编写多个爬虫**

一个程序里可以定义多个爬虫，此时最好在spider中定义item['come_from'] = 'A'，再定义多个pipeline。或者通过spider.name来判断数据是来自哪个爬虫。

```python
class MyspiderPipeline:
    def process_item(self, item, spider): # 这里的spider，其实就是爬虫里定义的
        if item['come_from'] == 'jd':
            item['hello'] == 'world'
		return item
    
class MyspiderPipeline:
    def process_item(self, item, spider): # 这里的spider，其实就是爬虫里定义的
        if spider.name == 'jd':
            item['hello'] == 'world'
		return item
```

### logging

![image-20210415150738701](python%20notes.assets/image-20210415150738701.png)

```python
import scrapy
import logging

logger = logging.getLogger(__name__) # 这样就可以获得正确的位置

class ItcastSpider(scrapy.Spider):
    name = 'itcast'
    allowed_domains = ['itcast.cn']
    start_urls = ['https://www.itcast.cn/channel/teacher.shtml']

    def parse(self, response):
        for i in range(10):
            item = {}
            item['come_from'] = 'itcast'
            logging.warning(item) # 汇报时间、warning、root（这个位置汇报的毫无意义）
            logger.warning(item) # 这样warning汇报的位置就是正确的了
        pass
    
# 如果在setting里面加一行，可以将日志文件保存到本地
LOG_FILE = './log.log' # 存到当前目录下的log.log中
```

### 翻页请求

```python
# 在parse函数里，最后添加下一页
next_url = response.xpath("//a[@id='next']/@href").extract_first()
if len(next_url) != 0:
    next_url = 'http://hr.tencent.com/' + next_url
    yield scrapy.Request(next_url, callback=self.parse2, meta={'item':item}, dont_filter=False) # 将url地址继续用self.parse的方法操作，如果是新的方法就要是self.parse2这样，meta方法可以把这边的数据传输到下一个parse里去，dont_filter表明不过滤，默认为False即过滤重复的url。有时候，比如爬股吧的时候，要反复的爬，就会设置成True。
    
def parse2(self, response):
    response.meta['item']
```

### selenium+scrapy

**settings**

```python
LOG_LEVEL = 'WARNING'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
ROBOTSTXT_OBEY = False

DOWNLOADER_MIDDLEWARES = {
'macro_spider.middlewares.JavaScriptMiddleware': 543, # 添加此行代码
# 该中间件将会收集失败的页面，并在爬虫完成后重新调度。（失败情况：临时问题，例如连接超时或者HTTP 500错误导致失败的页面）
'scrapy.downloadermiddlewares.retry.RetryMiddleware': 80,
# 该中间件提供了对request设置HTTP代理的支持。您可以通过在 Request 对象中设置 proxy 元数据来开启代理。
'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 100,
}

ITEM_PIPELINES = {
    'macro_spider.pipelines.MacroSpiderPipeline': 300,
}
```

**middlewares**

```python
import time
from selenium import webdriver
from scrapy.http import HtmlResponse

class JavaScriptMiddleware(object):
    def process_request(self, request, spider):
        print('开始渲染......')
        driver = webdriver.Chrome(executable_path=r'E:\终生学习\数据结构\chromedriver.exe')
        driver.get(request.url)
        time.sleep(1)
        js = "var q=document.documentElement.scrollTop=10000"
        driver.execute_script(js)  # 可执行js，模仿用户操作。此处为将页面拉至最底端。
        time.sleep(2)
        body = driver.page_source
        # print ("访问" + request.url)
        return HtmlResponse(driver.current_url, body=body, encoding='utf-8', request=request)
```



## XPath

- 解析XML的一种语言（HTML隶属于XML）；
- 大多数语言都有XPath；
- 除了XPath还有其他手段解析XML：beautifulsoup、lxml等。

**语法**：

- 层级：`/`直接子级、`//`跳级；
- 属性：`@`属性访问；
- 函数：`contain()`、`text()`等。

```python
//div[@class='xxx'] # //搜索所有子孙级，找div，[]是谓语，表示条件语句，@表示属性。
//div[@class='x']//span[contains(@class, 'xx')]
result = response.xpath("//div[@class='xxx']/text()").extract() # 提取里面的文字
```



## selenium

```python
from tqdm import tqdm,trange
from selenium import webdriver
from lxml import etree
import time
import pandas as pd

driver = webdriver.Chrome(executable_path=r'C:\Users\YANGQINGYUAN\Downloads\chromedriver.exe')

key_list = ['清源', '龙湖', '南岸'] 
href = []
shxy = []

for key in tqdm(key_list):
    url = "https://www.tianyancha.com/search?key={}".format(key)
    
    driver.get(url)
    time.sleep(2)
    temp = driver.find_element_by_xpath("//div[@id='search_company_0']//a[@class='name  ']")
    temp_url = temp.get_attribute('href')
    href.append(temp_url)
    driver.get(temp_url)
    time.sleep(2)
    temp = driver.find_element_by_xpath("//span[@class='copy-it info-need-copy _creditcode']").text
    shxy.append(temp)

resultdf = pd.DataFrame([key_list, href, shxy], index=['公司','网页','信用代码']).T
```



## RE

### 7种境界

regular expression

帮助你找到某种模式的字符串：是什么字符？重复多少次？在什么位置？有哪些额外的约束？

```python
import re
text = '麦叔身高：178，体重：168，学号：123456，密码：9527'
result = re.findall(tag, text) # 在text中找tag的模式

# level 1: 找到固定的字符串
result = re.findall(r'123456', text)

# level 2: 找到某一类字符
result = re.findall(r'\d', text) # 找到单个数字，\d表示任意数字，\D表示任意非数字的字符，\w表示非标点之外的任意字符，[1-5]只找1-5之内的数字，[高重号]找到[]中任意一个，[a-z]

# level 3：重复某一类字符
result = re.findall(r'\d+', text) # \d+是任意多个数字，这个返回178,168,123456,9527
result = re.findall(r'\d?', text) # \d?是1次或0次
result = re.findall(r'\d[1-5]', text) # \d[1-5]返回数字+1-5之间的组合，比如12,34,95
result = re.findall(r'\d{3}', text) # \d{3}抓后面3个
result = re.findall(r'\d{1,4}', text) # \d{3}抓后面1、2、3、4个都行
result = re.findall(r'\d{1,}', text) # \d{3}抓后面1个及以上都行
result = re.findall(r'\d{,8}', text) # \d{3}抓后面8个及以下都行

# level 4：组合level 2
result = re.findall(r'\d{4}-\d{8}', text) # 4位数字+8位数字 0514-87618156

# level 5：多种情况
result = re.findall(r'\d{4}-\d{8}|1\d{10}', text) # 4位数字+8位数字 0514-87618156 或 1后面的10位数字 15366867314

# level 6：限定位置
result = re.findall(r'^\d{4}-\d{8}|^1\d{10}', text) # ^表示开头
result = re.findall(r'\d{4}-\d{8}$|1\d{10}$', text) # $表示结尾

# level 7：内部限制
result = re.findall(r'(\w{3})(\1)', text) # ()表示分组，(\1)就表明后面这个组要和第一组一样
```

### 写RE的步骤

- 确定模式包含几个子模式；
- 各个部分的字符分类是什么；
- 各个子模式如何重复；
- 是否有外部位置限制；
- 是否有内部制约关系；

### 语法查询

- 字符类别
  - ![image-20210420140947472](python%20notes.assets/image-20210420140947472.png)
- 字符重复次数
  - ![image-20210420141053881](python%20notes.assets/image-20210420141053881.png)
- 组合模式
  - ![image-20210420141143828](python%20notes.assets/image-20210420141143828.png)
- 位置
  - ![image-20210420141437442](python%20notes.assets/image-20210420141437442.png)
- 分组
  - ![image-20210420142039036](python%20notes.assets/image-20210420142039036.png)
- 其他 findall(tag, text, flags=re.I)
  - ![image-20210420142208225](python%20notes.assets/image-20210420142208225.png)

### re的函数

```python
# 查找
re.search() # 只返回1个Match对象
re.match() # 必须从头开始匹配，也只返回1个Match对象
re.findall() # 返回所有字符串
re.finditer() # 返回Match对象迭代器 for i in re.finditer(): print(i)

# 替换
re.sub(tag, replace) # 替换，返回替换完的字符串
re.subn() # 替换完告诉我替换了多少个

# 分割
re.split()
```





# API

```python
# joinquant 付费数据
# 详见 https://www.joinquant.com/help/api/help?name=JQData

# Wind 付费数据
# 有代码生成器 从wind终端进去

# baostock 免费股票数据
# http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5

# tushare 免费股票数据

# csmar的代码处理
def num_to_code(x):
    try:
        x = (6-len(str(round(x)))) * '0' + str(round(x))
        if x[0:2] in ['00','30']:
            x = x + '.SZ'
        elif x[0:2] in ['60','68','T0']:
            x = x + '.SH'
    except:
        x = np.nan
    return x
dummydf['wind代码'] = dummydf['wind代码'].apply(lambda x: num_to_code(x))
temp = dummydf['wind代码'].str.lstrip('0123456789.')
temp.isin(['SZ','SH']).value_counts()
temp2 = dummydf['wind代码'][~temp.isin(['SZ','SH'])]
```

## Ta-lib

```python
import talib as ta

# ta.MAX / ta.MIN 返回过去timeperiod中的最大值/最小值
upperband = ta.MAX(self.high[:-1], timeperiod = self.N2) # 不包括今天，只看过去，否则如果今天是最高最低，就不存在击破的情况了
upperlimit = upperband.iloc[-1]
lowerband = ta.MIN(self.low[:-1], timeperiod = self.N2)
lowerlimit = lowerband.iloc[-1]

# ATR 单日最大振幅（high - low - close_t-1）的移动平均
```

# Others

## pyautogui

```python
import pyautogui as pg

# 返回静态鼠标位置
pg.position()

# 模拟鼠标操作
pg.click() # 单击
pg.click(button='right') # 右键 
pg.doubleClick() # 双击
pg.moveTo(100,200,2) # 用2秒的啥时间移动到（100,200）
pg.dragTo(300,400,2,button='left') # 用左键拖拽到（300,400）的位置

# 模拟键盘操作
pg.press('enter') # 按键
pg.press('left')
pg.press('ctrl')
pg.hotkey('ctrl','c') # 快捷键
pg.KeyDown('ctrl') # 按住不动
pg.KeyUp() # 松开键位
pg.PAUSE = 1 # 每隔1秒操作一次
pg.write('fuck python!',interval=0.3) # 输入内容
```

## Pyinstaller

```python
pyinstaller -F xx.py
```

# IDE

## IPython

### 重载模块

```额python
# 当你修改完你的一个模块后，其他脚本中引用这个模块的时候还未更新，需要重新加载
import importlib
importlib.reload(some_lib)

# 另一种方法
dreload(some_lib) # 在ipython中输入
```

### 魔法键与快捷键

```python
%matplotlib qt # 新建窗口可以操作图形
ctrl + D # 删除整行
shift + M # 合并单元格
esc + F # 查找与替换
```

## pycharm

```python
ctrl + shift + F10 # 运行
cls # 清理日志
```

# Make a PDF

## 导入模块

```python
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus import PageBreak
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame
from reportlab.lib.units import cm
```

## 搭建框架

```python
# 定义模板
class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        self.allowSplitting = 0
        BaseDocTemplate.__init__(self, filename, **kw)
        template = PageTemplate('normal', [Frame(2.5*cm, 2.5*cm, 15*cm, 25*cm, id='F1')])
        self.addPageTemplates(template)
    
    def afterFlowable(self, flowable):
        "Registers TOC entries."
        if flowable.__class__.__name__ == 'Paragraph':
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'title1':
                level = 0
            elif style == 'title2':
                level = 1
            elif style == 'title3':
                level = 2
            else:
                return
            E = [level, text, self.page]
            #if we have a bookmark name append that to our notify data
            bn = getattr(flowable,'_bookmarkName',None)
            if bn is not None:
                E.append(bn)
            self.notify('TOCEntry', tuple(E))  

# 注册字体
pdfmetrics.registerFont(TTFont('hwzs', 'STZHONGS.ttf'))

# 表格格式，三线表
ts = [('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('FONT', (0,0), (-1,-1), 'hwzs'),
    ('LINEABOVE', (0,0), (-1,0), 1, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 1, colors.black)]
        
# 定义样式
stylesheet = getSampleStyleSheet()
stylesheet.add(
    ParagraphStyle(name='body',
                   fontName="hwzs",
                   fontSize=10,
                   textColor='black',
                   leading=15,                # 行间距
                   spaceBefore=15,             # 段前间距
                   spaceAfter=15,             # 段后间距
                   leftIndent=0,              # 左缩进
                   rightIndent=0,             # 右缩进
                   firstLineIndent=20,        # 首行缩进，每个汉字为10
                   alignment=TA_JUSTIFY,      # 对齐方式
                   ))
stylesheet.add(
    ParagraphStyle(name='title1',
                   fontName="hwzs",
                   fontSize=16,
                   textColor='black',
                   leading=15,                # 行间距
                   spaceBefore=15,             # 段前间距
                   spaceAfter=15,             # 段后间距
                   leftIndent=0,              # 左缩进
                   rightIndent=0,             # 右缩进
                   firstLineIndent=0,        # 首行缩进，每个汉字为10
                   alignment=TA_JUSTIFY    # 对齐方式
                   ))
stylesheet.add(
                ParagraphStyle(name='title2',
                   fontName="hwzs",
                   fontSize=14,
                   textColor='black',
                   leading=15,                # 行间距
                   spaceBefore=15,             # 段前间距
                   spaceAfter=15,             # 段后间距
                   leftIndent=0,              # 左缩进
                   rightIndent=0,             # 右缩进
                   firstLineIndent=0,        # 首行缩进，每个汉字为10
                   alignment=TA_JUSTIFY   # 对齐方式
                   ))
stylesheet.add(
            ParagraphStyle(name='title3',
                   fontName="hwzs",
                   fontSize=12,
                   textColor='black',
                   leading=15,                # 行间距
                   spaceBefore=15,             # 段前间距
                   spaceAfter=15,             # 段后间距
                   leftIndent=0,              # 左缩进
                   rightIndent=0,             # 右缩进
                   firstLineIndent=0,        # 首行缩进，每个汉字为10
                   alignment=TA_JUSTIFY    # 对齐方式
                   ))
stylesheet.add(
    ParagraphStyle(name='title0',
                   fontName="hwzs",
                   fontSize=20,
                   textColor='black',
                   leading=15,                # 行间距
                   spaceBefore=15,             # 段前间距
                   spaceAfter=15,             # 段后间距
                   leftIndent=0,              # 左缩进
                   rightIndent=0,             # 右缩进
                   firstLineIndent=0,        # 首行缩进，每个汉字为10
                   alignment=1    # 对齐方式
                   ))

h0 = stylesheet['title0']
h1 = stylesheet['title1']
h2 = stylesheet['title2']
h3 = stylesheet['title3']
body = stylesheet['body'] # 正文

# 添加超链接
def doheading(text,sty):
    from hashlib import sha1
    #create bookmarkname
    bn=sha1((text+sty.name).encode('utf-8')).hexdigest()
    #modify paragraph text to include an anchor point with name bn
    h=Paragraph(text+'<a name="%s"/>' % bn,sty)
    #store the bookmark name on the flowable so afterFlowable can see this
    h._bookmarkName=bn
    story.append(h)
    
# 初始化内容
story = []

# 大标题
text = "{}复盘报告".format(today.strftime('%Y%m%d'))
story.append(Paragraph(text, h0))
# 目录
toc = TableOfContents()

toc.levelStyles = [
    PS(fontName='hwzs', fontSize=16, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, leading=16),
    PS(fontName='hwzs',fontSize=14, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=5, leading=12),
     PS(fontName='hwzs',fontSize=12, name='TOCHeading3', leftIndent=60, firstLineIndent=-20, spaceBefore=5, leading=12),   
]
story.append(toc)

# 最大的标题

text2 = "{}复盘报告".format(today.strftime('%Y%m%d'))
story.append(PageBreak())
doheading('标题1', h1)
doheading('我是二级标题', h2)
doheading('我是三级标题', h3)
story.append(Paragraph(text, body))

# 将内容输出到PDF中
doc = MyDocTemplate("【{}】每日复盘.pdf".format(today.strftime('%Y%m%d')))
doc.multiBuild(story)
```

## 换页符

```python
# 下一页
from reportlab.platypus import PageBreak
story.append(PageBreak())

```
