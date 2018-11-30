# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:27:55 2018

@author: zwe
"""
#https://github.com/691175002/BLPInterface

import blpinterface.blp_interface as blp
import datetime
#blp = blp.BLPInterface()
#a = blp.referenceRequest(['CNR CN Equity', 'CP CN Equity'], ['SECURITY_NAME_REALTIME', 'LAST_PRICE'])

#aapl = Price("AAPL US")
#a = aapl.price
para = {'adjustmentFollowDPDF':True,'nonTradingDayFillOption':"ALL_CALENDAR_DAYS"}
b = blp.historicalRequest('AAPL US Equity', ['PX_LAST'], '19911202', '20180911',**para)
c = a.join(b)

para1 = {'nonTradingDayFillOption':"ALL_CALENDAR_DAYS"}

a = blp.historicalRequest('AAPL US Equity', ['PX_LAST','BEST_EPS','IS_EPS','EARN_YLD','EARN_YLD_HIST','BEST_PE_RATIO'], '20160101', '20180912',**para1)
a1 = blp.historicalRequest('AAPL US Equity', ['PX_LAST','BEST_EPS','IS_EPS','EARN_YLD','EARN_YLD_HIST','BEST_PE_RATIO'], '20160101', '20180912', overrides = {'BESTFPERIOD_OVERRIDE' : '1BF'})


a = blp.historicalRequest('AAPL US Equity', ['PX_LAST','BEST_EPS','IS_EPS','EARN_YLD','EARN_YLD_HIST'], '20160101', '20180912',**para)

a = blp.historicalRequest('NFP TCH Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '20161231', '20180906')

a = blp.historicalRequest('RSAOFURN Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '20120101', '20181008')

b = blp.historicalRequest('RSAOAUTO Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '20111231', '20180924')


b = blp.historicalRequest('LEI YOY Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '19600101', '20180924')

c = blp.historicalRequest(['USGG10YR Index','USGG2YR Index','USYC2Y10 Index'], ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '19600101', '20180924')

d = c.ffill()


b = blp.historicalRequest('rsbaauto Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '20120101', '20181015')
#b = mg.Lv(b,12).all


mgr = blp.BLPInterface()
today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
df_eps = mgr.historicalRequest('SPX Index', 'INDX_WEIGHTED_EST_ERN', datetime.datetime(today.year-5, today.month, today.day), today,**para1, overrides = {'BESTFPERIOD_OVERRIDE' : '1BF'})

a_eps = mgr.historicalRequest('AAPL US Equity', 'BEST_EPS', datetime.datetime(today.year-5, today.month, today.day), today,**para1, overrides = {'BESTFPERIOD_OVERRIDE' : '1BF'})

#### change change #######
###
#
#
#

###########
