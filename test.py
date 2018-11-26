# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:24:29 2018

@author: zwe
"""
import pandas as pd

import sys; sys.path.append('M:\\python\\py35\\lib')
import LoadData_Wen3 as lw
import MetricsGenerator as mg
import Wen_Model as wm
import Equity_Model as em
ticker ='CMG'
security = em.Security(ticker+' US', sdate =dt.date(2010,1,1))
sss = security.get_hist_consensus('SAMESTORESALES')
st = security.get_hist_consensus('ST_END')
    

sql = """select a.clicktype,b.date,count(a.*),count(distinct a.session_id) from jumpshot.stablePanel_nflx a right join rt_week_ending b on a.date = b.date where b.date>='20140101' and b.date<=getdate() group by 1,2 order by 1,2;"""
data = lw.get_aws_sql(sql)
data.columns = ['clicktype','date','row','session']
row_pvt = pd.pivot_table(data,'row','date','clicktype')
session_pvt = pd.pivot_table(data,'session','date','clicktype')
session_pvt.columns = [col+'_session' for col in session_pvt]
for col in row_pvt.columns:
    temp = row_pvt[[col]].join(session_pvt[[col+'_session']])
    temp.plot(secondary_y=[col+'_session'],title=col)
    
col = 'AJAX'
temp = row_pvt[[col]].join(session_pvt[[col+'_session']])

for tbl in ['amzn','wen_adsk_total_temp','wen_adsk_total_dedup','wen_adsk_stable_temp','wen_adsk_stable_dedup','nflx_2_bank_stats','nflx_2_card_stats']:
    lw.drop_aws_table(tbl)
    
def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
   pivotvalue = alist[first]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp

   temp = alist[first]
   alist[first] = alist[rightmark]
   alist[rightmark] = temp


   return rightmark

alist = [54,26,93,17,77,31,44,55,20]
quickSort(alist)
print(alist)


def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)

alist = [54,26,93,17,77,31,44,55,20]
mergeSort(alist)
print(alist)



def binarySearch(alist, item):
    if len(alist) == 0:
        return False
    else:
         midpoint = len(alist)//2
         if alist[midpoint]==item:
             return True
         else:
	          if item<alist[midpoint]:
	            return binarySearch(alist[:midpoint],item)
	          else:
	            return binarySearch(alist[midpoint+1:],item)
	
testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42,]
print(binarySearch(testlist, 3))
print(binarySearch(testlist, 13))


from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
import pylab

# Problem data.
n = 4
S = matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ],
            [ 6e-3,  1e-2,  0.0,     0.0 ],
            [-4e-3,  0.0,   2.5e-3,  0.0 ],
            [ 0.0,   0.0,   0.0,     0.0 ]])
pbar = matrix([.12, .10, .07, .03])
G = matrix(0.0, (n,n))
G[::n+1] = -1.0
h = matrix(0.0, (n,1))
A = matrix([[ 4e-2,  6e-3, -4e-3,    -1.0],
            [ 6e-3,  1e-2,  0.0,     0.5 ]])
b = matrix([[1.0],[0]])

# Compute trade-off.
N = 100
mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
returns = [ dot(pbar,x) for x in portfolios ]
risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

# Plot trade-off curve and optimal allocations.
pylab.figure(1, facecolor='w')
pylab.plot(risks, returns)
pylab.xlabel('standard deviation')
pylab.ylabel('expected return')
pylab.axis([0, 0.2, 0, 0.15])
pylab.title('Risk-return trade-off curve (fig 4.12)')
pylab.yticks([0.00, 0.05, 0.10, 0.15])

pylab.figure(2, facecolor='w')
c1 = [ x[0] for x in portfolios ]
c2 = [ x[0] + x[1] for x in portfolios ]
c3 = [ x[0] + x[1] + x[2] for x in portfolios ]
c4 = [ x[0] + x[1] + x[2] + x[3] for x in portfolios ]
pylab.fill(risks + [.20], c1 + [0.0], '#F0F0F0')
pylab.fill(risks[-1::-1] + risks, c2[-1::-1] + c1, facecolor = '#D0D0D0')
pylab.fill(risks[-1::-1] + risks, c3[-1::-1] + c2, facecolor = '#F0F0F0')
pylab.fill(risks[-1::-1] + risks, c4[-1::-1] + c3, facecolor = '#D0D0D0')
pylab.axis([0.0, 0.2, 0.0, 1.0])
pylab.xlabel('standard deviation')
pylab.ylabel('allocation')
pylab.text(.15,.5,'x1')
pylab.text(.10,.7,'x2')
pylab.text(.05,.7,'x3')
pylab.text(.01,.7,'x4')
pylab.title('Optimal allocations (fig 4.12)')
pylab.show()

"""
QuantConnect Logo Welcome to The QuantConnect Research Page!

Refer to this page for documentation https://www.quantconnect.com/docs#Introduction-to-Jupyter

Contribute to this file https://github.com/QuantConnect/Research/tree/master/Notebooks

Mean Variance Portfolio Optimization
In this research we will demonstrate Markowitz portfolio optimization model in Python. Then we use a plot to show the efficient frontier, the optimal weight and the minimum variance weight.
"""
#%matplotlib inline
from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Jupyter")
AddReference("QuantConnect.Indicators")
from System import *
from QuantConnect import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from QuantConnect.Jupyter import *
from QuantConnect.Indicators import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
qb = QuantBook()
class PortfolioOptimization:
    """ 
    Description: 
        This class shows how to set up a basic asset allocation problem
        that uses mean-variance portfolio optimization to estimate efficient
        portfolio and plot the efficient frontier and capital market line
    Args:
       log_return(pandas.DataFrame): The log return for assets in portfolio
                                    index: date
                                    columns: symbols
                                    value: log return series                                    
       risk_free_rate(float): The risk free rate 
       num_assets(int): The number of assets in portfolio 
        
    """


    def __init__(self, log_return, risk_free_rate, num_assets):
        self.log_return = log_return
        self.risk_free_rate = risk_free_rate
        self.n = num_assets 

    def annual_port_return(self, weights):
        ''' calculate the annual return of portfolio ''' 
        
        return np.sum(self.log_return.mean() * weights) * 252

    def annual_port_vol(self, weights):
        ''' calculate the annual volatility of portfolio ''' 
        
        return np.sqrt(np.dot(weights.T, np.dot(self.log_return.cov() * 252, weights)))

    def mc_mean_var(self):
        ''' apply monte carlo method to search for the feasible region of return '''
        
        returns = []
        vols = []
        for i in range(5000):
            weights = np.random.rand(self.n)
            weights /= np.sum(weights)
            returns.append(self.annual_port_return(weights))
            vols.append(self.annual_port_vol(weights))
        return returns, vols

    def min_func(self, weights):
        return - self.annual_port_return(weights) / self.annual_port_vol(weights)

    def opt_portfolio(self):
        ''' maximize the sharpe ratio to find the optimal weights ''' 
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(self.n))
        opt = minimize(self.min_func,
                       np.array(self.n * [1. / self.n]),
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        opt_weights = opt['x']
        opt_return = self.annual_port_return(opt_weights)
        opt_volatility = self.annual_port_vol(opt_weights)

        return opt_weights, opt_return, opt_volatility

    def min_var_portfolio(self):
        ''' find the portfolio with minimum volatility '''
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(self.n))
        opt = minimize(self.annual_port_vol,
                       np.array(self.n * [1. / self.n]),
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        min_vol_weights = opt['x']
        min_vol_return = self.annual_port_return(min_vol_weights)
        min_volatility = self.annual_port_vol(min_vol_weights)
        return min_vol_weights, min_vol_return, min_volatility

    def efficient_frontier(self, mc_returns):
        ''' calculate the efficient frontier ''' 
        
        target_return = np.linspace(min(mc_returns), max(mc_returns) + 0.05, 100)
        target_vol = []
        bnds = tuple((0, 1) for x in range(self.n))
        for i in target_return:
            cons = ({'type': 'eq', 'fun': lambda x: self.annual_port_return(x) - i},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            opt = minimize(self.annual_port_vol,
                           np.array(self.n * [1. / self.n]),
                           method='SLSQP',
                           bounds=bnds,
                           constraints=cons)
            target_vol.append(opt['fun'])
        return target_vol, target_return

    def plot(self):
        mc_returns, mc_vols = self.mc_mean_var()
        target_vol, target_return = self.efficient_frontier(mc_returns)
        opt_weights, opt_return, opt_volatility = self.opt_portfolio()
        min_vol_weights, min_vol_return, min_volatility = self.min_var_portfolio()

        plt.figure(figsize=(15, 8))
        # plot the possible mean-variance portfolios with monte carlo simulation
        excess_return = [i - self.risk_free_rate for i in mc_returns]
        plt.scatter(mc_vols, mc_returns, c=np.array(excess_return) / np.array(mc_vols), cmap=cm.jet, marker='.')
        plt.grid()
        plt.xlabel('standard deviation(annual)')
        plt.ylabel('expected return(annual)')
        plt.colorbar(label='Sharpe ratio')

        # plot the efficient frontier
        plt.scatter(target_vol, target_return, c=np.array(target_return) / np.array(target_vol), marker='.', cmap=cm.jet)

        # mark the optimal portfolio with green star
        plt.scatter(opt_volatility, opt_return, marker='*', s=200, c='g', label = 'optimal portfolio')

        # mark the min volatility portfolio with purple star
        plt.scatter(min_volatility, min_vol_return, marker='*', s=200, c='m', label = 'min volatility portfolio')

        # plot the capital market line with black
        cml_x = np.linspace(0.0, 0.3)
        cml_slope = (opt_return - self.risk_free_rate) / opt_volatility
        plt.plot(cml_x, self.risk_free_rate + cml_slope * cml_x, lw=1.5, c='k', label = 'capital market line')
        plt.legend()
np.random.seed(123)
symbols = ['CCE','AAP','AAPL','GOOG','IBM','AMZN'] 
for i in symbols:
    qb.AddEquity(i, Resolution.Daily)
history = qb.History(500, Resolution.Daily)
data = {}
for i in symbols:
    if i in history.index.levels[0]:
        data[i] = history.loc[i]['close'] 
df_price = pd.DataFrame(data,columns=data.keys()) 
# calculate the log return series for each stock
log_return = np.log(df_price / df_price.shift(1)).dropna()
# plot the distribution of log return for each stocks
log_return.hist(bins=50, figsize=(15, 12))
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fa63603f290>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fa635c71810>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x7fa635af8810>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fa635a69250>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x7fa6358f0290>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fa635855990>]], dtype=object)

# create the portfolio object 'a' with given symbols
a = PortfolioOptimization(log_return, 0, len(symbols))
opt_portfolio = a.opt_portfolio()
opt_portfolio[0]
array([  2.12741122e-01,   1.76385211e-01,   2.25515892e-01,
         1.74667680e-15,   6.49128202e-02,   3.20444954e-01])
a.plot()

# print out the return, volatility, Sharpe ratio and weights of the optimal portfolio
df = pd.DataFrame({'return':opt_portfolio[1], 'volatility':opt_portfolio[2]},
                  index=['optimal portfolio'])
df
return	volatility
optimal portfolio	0.162502	0.153377



from decorators import debug, timer

@timer
def waste_time(self, num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(self.max_num)])
            
waste_time(999)

def entry_exit(f):
    def new_f():
        print("Entering", f.__name__)
        f()
        print("Exited", f.__name__)
    return new_f

@entry_exit
def func1():
    print("inside func1()")

@entry_exit
def func2():
    print("inside func2()")

func1()
func2()
print(func1.__name__)
print(func2.__name__)