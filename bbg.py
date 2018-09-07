# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:27:55 2018

@author: zwe
"""
#https://github.com/691175002/BLPInterface
import blpinterface.blp_interface as blp

blp = blp.BLPInterface()

blp.historicalRequest('BMO CN Equity', 'PX_LAST', '20141231', '20150131')

a = blp.referenceRequest(['CNR CN Equity', 'CP CN Equity'], ['SECURITY_NAME_REALTIME', 'LAST_PRICE'])

a = blp.historicalRequest('BMO CN Equity', ['PX_LAST','BEST_EPS','IS_EPS','EARN_YLD'], '20161231', '20180906')

a = blp.historicalRequest('NFP TCH Index', ['PX_LAST','BN_SURVEY_MEDIAN','BN_SURVEY_AVERAGE'], '20161231', '20180906')