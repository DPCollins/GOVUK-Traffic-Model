from __future__ import division

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import pagination
import os
import sys
import csv
import numpy as np
import pandas as pd
import pandas.io.ga as ga
from pandas import *
from time import sleep
#import oct2py as op
from datetime import timedelta
from decimal import Decimal
import math
import pdb
from collections import Counter
from decimal import *
import csv
import time
import imp
from dateutil import parser
import math
import pymongo
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
import MySQLdb
from pandas.io import sql
import scipy.sparse as sps
from sklearn.feature_extraction import DictVectorizer
import neurolab as nl

# plt.close('all')

def Trial(): 

	t = datetime.today()

	yesterday = t - timedelta(0)
	dbyy = t - timedelta(90)

	start  = dbyy.strftime('%Y-%m-%d')
	today = t.strftime('%Y-%m-%d %H:00')
	# end = yesterday.strftime('%Y-%m-%d')

	start = "2014-06-07"
	end = "2014-06-10"

	top100 = [u'/jobsearch', u'/search', u'/', u'/tax-disc', u'/renew-adult-passport', u'/student-finance-register-login', u'/visas-immigration', u'/driving-transaction-finished', u'/browse/abroad/passports', u'/apply-uk-visa', u'/browse/driving', u'/check-uk-visa', u'/get-a-passport-urgently', u'/apply-renew-passport', u'/government/organisations/uk-visas-and-immigration', u'/book-practical-driving-test', u'/bank-holidays', u'/change-date-practical-driving-test', u'/government/organisations/driver-and-vehicle-licensing-agency', u'/contact-jobcentre-plus', u'/book-a-driving-theory-test', u'/browse/driving/driving-licences', u'/benefits-calculators', u'/check-uk-visa/y', u'/jobseekers-allowance/how-to-claim', u'/national-minimum-wage-rates', u'/tax-credits-calculator', u'/browse/benefits/tax-credits', u'/browse/benefits', u'/change-address-driving-licence', u'/contact-the-dvla', u'/browse/working/finding-job', u'/calculate-state-pension', u'/passport-fees', u'/browse/working', u'/contact/hm-revenue-customs/tax-credits-enquiries', u'/overseas-passports', u'/track-passport-application', u'/renew-driving-licence', u'/browse/abroad', u'/get-vehicle-information-from-dvla', u'/apply-first-provisional-driving-licence', u'/student-finance/overview', u'/browse/driving/car-tax-discs', u'/general-visit-visa', u'/apply-online-to-replace-a-driving-licence', u'/government/organisations/hm-passport-office', u'/check-mot-status', u'/uk-border-control', u'/get-a-child-passport', u'/practise-your-driving-theory-test', u'/renewing-your-tax-credits-claim', u'/renewtaxcredits', u'/calculate-state-pension/y', u'/student-finance', u'/photos-for-passports', u'/contact-student-finance-england', u'/visa-processing-times', u'/foreign-travel-advice', u'/jobseekers-allowance', u'/contact', u'/browse/education/student-finance', u'/calculate-vehicle-tax-rates', u'/find-a-visa-application-centre', u'/working-tax-credit', u'/renew-driving-licence-at-70', u'/passport-advice-line', u'/call-charges', u'/overseas-passports/y', u'/countersigning-passport-applications', u'/government/topical-events/sexual-violence-in-conflict', u'/how-the-post-office-check-and-send-service-works', u'/visa-fees', u'/government/organisations', u'/browse/driving/learning-to-drive', u'/browse/working/state-pension', u'/vehicle-tax-rate-tables', u'/get-a-child-passport/your-childs-first-passport', u'/calculate-state-pension/y/age', u'/make-a-sorn', u'/jobseekers-allowance/what-youll-get', u'/general-visit-visa/apply', u'/contact/govuk/anonymous-feedback/thankyou', u'/browse/citizenship/citizenship', u'/general-visit-visa/documents-you-must-provide', u'/jobseekers-allowance/overview', u'/uk-border-control/before-you-leave-for-the-uk', u'/government/organisations/foreign-commonwealth-office', u'/government/collections/national-curriculum', u'/government/organisations/ministry-of-defence', u'/ips-regional-passport-office', u'/hand-luggage-restrictions/overview', u'/jobseekers-allowance/eligibility', u'/register-to-vote', u'/disclosure-barring-service-check/overview', u'/browse/benefits/jobseekers-allowance', u'/dvlaforms', u'/tier-4-general-visa', u'/student-finance/loans-and-grants']

	max_results = 5e7
	metrics = ['pageviews']
	dimensions = ['pagePath', 'hour', 'date']
	dim = ['date', 'hour']
	filters = ['pagePath=='+top100[97]]

	###############
	#Find Top 100 pages by pageviews - (pv fine in this case rather than upv)

	# df = ga.read_ga(metrics, 
	# 					dim, 
	# 					start_date = start,
	# 					end_date = end,
	# 					token_file_name = 'static/token/analytics.dat',
	# 					secrets = 'static/token/client_secrets.json',
	# 					account_id = '26179049',
	# 					max_results=max_results,        
	#                     chunksize=5000
	# 			   )
	# df1b = pd.concat([x for x in df])
	# df1c = df1b.sort(columns=['pageviews'], ascending=0)

	df1 = ga.read_ga(metrics,
						 dimensions = dim,
						 start_date = dbyy, 
						 end_date = yesterday, 
						 parse_dates=[['date', 'hour']],
						 token_file_name = 'static/token/analytics.dat',
						 secrets = 'static/token/client_secrets.json',
						 account_id = '26179049'
	# 					 filters = filters
						 )

	##################### 48 MAX LAG ##############################

	ind = []

	for i in range(48, len(df1)):

		lag = [1,2,3,24,48]
		lagx = list(i - np.array(lag))
	
		Train = df1.ix[lagx]
		Target = df1.ix[i]

		TT = Train.T
		TT.columns = [1,2,3,24,48]
		TT['Target'] = Target['pageviews']
		ind.append(TT)

	rng = date_range(df1.index[48], df1.index[len(df1)-1], freq='H')
	Set = ind[0].append(ind[1:])
	Set.index = rng
	SetT = Set.ix[:today][:-1]
	print SetT

	##################### 7 day trial ##############################

	# ind = []
	# 
	# for i in range(168, len(df1)):
	# 
	# 	lag = [1,2,3,24,48, 168]
	# 	lagx = list(i - np.array(lag))
	# 	
	# 	Train = df1.ix[lagx]
	# 	Target = df1.ix[i]
	# 
	# 	TT = Train.T
	# 	TT.columns = [1,2,3,24,48, 168]
	# 	TT['Target'] = Target['pageviews']
	# 	ind.append(TT)
	# 
	# rng = date_range(df1.index[168], df1.index[len(df1)-1], freq='H')
	# Set = ind[0].append(ind[1:])
	# Set.index = rng
	# SetT = Set.ix[:today][:-1]
	# print SetT


	#################################################
	TrainSamp = 0.8
	feats = 5
	TS = int(np.round(TrainSamp*len(SetT)))

	X_Train = SetT[SetT.columns[0:feats]].head(TS)
	Y_Train = SetT['Target'].head(TS)

	X_Test = SetT[SetT.columns[0:feats]].ix[TS:]
	Y_Test = SetT['Target'].ix[TS:]

	X_Train = X_Train.replace(0,1)
	X_Test = X_Test.replace(0,1)
	Y_Train = Y_Train.replace(0,1)
	Y_Test = Y_Test.replace(0,1)

	print 50*'-'
	print "Random Forest Regression"
	print 50*'-'
	rf = RandomForestRegressor(n_estimators=500, max_features=feats)
	rf.fit(X_Train, Y_Train)
	PredRF = rf.predict(X_Test)
	scoreRF = r2_score(Y_Test,PredRF)
	MSE = np.mean(np.square(Y_Test-PredRF))
	print 'R2 Score = ', scoreRF 
	print 'MSE = ',  MSE

	Res = pd.DataFrame(columns=['res'], index = (range(0,len(Y_Test))))
	resid = Y_Test-PredRF
	Res['res'] = resid.values

	TSDU = Res['res'].mean()+3*Res['res'].std()
	TSDL = Res['res'].mean()-3*Res['res'].std()
	tsdP = Res[Res['res']>(Res['res'].mean()+3*Res['res'].std())]
	tsdN = Res[Res['res']<(Res['res'].mean()-3*Res['res'].std())]


	plt.figure(2)
	plt.plot(Y_Test, Y_Test)
	plt.scatter(Y_Test,PredRF, s=40, alpha=0.5, c='r')

	##############################
	plt.figure(1)

	plt.subplot(2, 2, 1)
	plt.plot(range(0,len(Y_Test)), Y_Test)
	plt.scatter(range(0,len(Y_Test)), PredRF, s=70, alpha=0.5, c='r')
	plt.xlim([0,len(Y_Test)])
	plt.title("Random Forest Model")


	plt.subplot(2,2,3)

	plt.plot(range(0,len(Y_Test)),resid)
	plt.plot(range(0,len(Y_Test)), [TSDU]*len(Y_Test), c='r')
	plt.plot(range(0,len(Y_Test)), [TSDL]*len(Y_Test), c='r')
	plt.xlim([0,len(Y_Test)])

	###############################





	AP = len(tsdP) + len(tsdN)

	print 50*'-'  
	print "PAGE: " + str(filters[0])
	print 50*'-'
	print "RANDOM FOREST Number of Anomalous Points is: %i" % AP
	print "%% Anomalous points is: %.1f" % (100*AP/len(Y_Test)) +'%'

	#################################################################

	print 50*'-'
	print "Linear Model"
	print 50*'-'
	clf = linear_model.LinearRegression()
	clf.fit(np.log(X_Train), np.log(Y_Train))
	LMPred = clf.predict(np.log(X_Test))
	scoreLM = r2_score(np.log(Y_Test),LMPred)
	MSELM =  np.mean(np.square(Y_Test-np.exp(LMPred)))
	print "R2 Score = ", scoreLM
	print "MSE = ", MSELM


	###########################################################

	print 50*'-'
	print "LASSO Regression"
	print 50*'-'


	################################################################################
	# Lasso with path and cross-validation using LassoCV path
	from sklearn.linear_model import LassoCV

	lasso_cv = LassoCV()

	y_ = lasso_cv.fit(X_Train, Y_Train).predict(X_Test)
	# .predict(X_Test)

	print "Optimal regularization parameter  = %s" % lasso_cv.alpha_

	# Compute explained variance on test data
	print "r^2 on test data : %f" % (1 - np.linalg.norm(Y_Test - y_)**2
										  / np.linalg.norm(Y_Test)**2)

	print "LASSO MSE: ", np.mean(np.square((Y_Test-(y_))))

	Res2 = pd.DataFrame(columns=['res'], index = (range(0,len(Y_Test))))
	resid2 = Y_Test-y_
	Res2['res'] = resid2.values

	TSDU2 = Res2['res'].mean()+3*Res2['res'].std()
	TSDL2 = Res2['res'].mean()-3*Res2['res'].std()
	tsdP2 = Res2[Res2['res']>(Res2['res'].mean()+3*Res2['res'].std())]
	tsdN2 = Res2[Res2['res']<(Res2['res'].mean()-3*Res2['res'].std())]

	AP = len(tsdP2) + len(tsdN2)


	print 50*'-'  
	print "PAGE: " + str(filters[0])
	print 50*'-'
	print "LASSP Number of Anomalous Points is: %i" % AP
	print "%% Anomalous points is: %.1f" % (100*AP/len(Y_Test)) +'%'

	# plt.figure(3) #################################################

	plt.subplot(2, 2, 2)
	plt.plot(range(0,len(Y_Test)), Y_Test)
	plt.scatter(range(0,len(Y_Test)), y_, s=70, alpha=0.5, c='r')
	plt.xlim([0,len(Y_Test)])
	plt.title('LASSO Model')


	plt.subplot(2,2,4)

	plt.plot(range(0,len(Y_Test)),resid2)
	plt.plot(range(0,len(Y_Test)), [TSDU2]*len(Y_Test), c='r')
	plt.plot(range(0,len(Y_Test)), [TSDL2]*len(Y_Test), c='r')
	plt.xlim([0,len(Y_Test)])
	
	

####################################################################

#Hourprediction

Store = []

top100 = [None, u'/jobsearch', u'/search', u'/', u'/tax-disc', u'/renew-adult-passport', u'/student-finance-register-login', u'/visas-immigration', u'/driving-transaction-finished', u'/browse/abroad/passports', u'/apply-uk-visa', u'/browse/driving', u'/check-uk-visa', u'/get-a-passport-urgently', u'/apply-renew-passport', u'/government/organisations/uk-visas-and-immigration', u'/book-driving-test', u'/bank-holidays', u'/change-date-practical-driving-test', u'/government/organisations/driver-and-vehicle-licensing-agency', u'/contact-jobcentre-plus', u'/book-theory-test', u'/browse/driving/driving-licences', u'/benefits-calculators', u'/check-uk-visa/y', u'/jobseekers-allowance/how-to-claim', u'/national-minimum-wage-rates', u'/tax-credits-calculator', u'/browse/benefits/tax-credits', u'/browse/benefits', u'/change-address-driving-licence', u'/contact-the-dvla', u'/browse/working/finding-job', u'/calculate-state-pension', u'/passport-fees', u'/browse/working', u'/government/organisations/hm-revenue-customs/contact/tax-credits-enquiries', u'/overseas-passports', u'/track-passport-application', u'/renew-driving-licence', u'/browse/abroad', u'/get-vehicle-information-from-dvla', u'/apply-first-provisional-driving-licence', u'/student-finance/overview', u'/browse/driving/car-tax-discs', u'/general-visit-visa', u'/apply-online-to-replace-a-driving-licence', u'/government/organisations/hm-passport-office', u'/check-mot-status', u'/uk-border-control', u'/get-a-child-passport', u'/practise-your-driving-theory-test', u'/renewing-your-tax-credits-claim', u'/renewtaxcredits', u'/calculate-state-pension/y', u'/student-finance', u'/photos-for-passports', u'/contact-student-finance-england', u'/visa-processing-times', u'/foreign-travel-advice', u'/jobseekers-allowance', u'/contact', u'/browse/education/student-finance', u'/calculate-vehicle-tax-rates', u'/find-a-visa-application-centre', u'/working-tax-credit', u'/renew-driving-licence-at-70', u'/passport-advice-line', u'/call-charges', u'/overseas-passports/y', u'/countersigning-passport-applications', u'/government/topical-events/sexual-violence-in-conflict', u'/how-the-post-office-check-and-send-service-works', u'/visa-fees', u'/government/organisations', u'/browse/driving/learning-to-drive', u'/browse/working/state-pension', u'/vehicle-tax-rate-tables', u'/get-a-child-passport/your-childs-first-passport', u'/calculate-state-pension/y/age', u'/make-a-sorn', u'/jobseekers-allowance/what-youll-get', u'/general-visit-visa/apply', u'/contact/govuk/anonymous-feedback/thankyou', u'/browse/citizenship/citizenship', u'/general-visit-visa/documents-you-must-provide', u'/jobseekers-allowance/overview', u'/uk-border-control/before-you-leave-for-the-uk', u'/government/organisations/foreign-commonwealth-office', u'/government/collections/national-curriculum', u'/government/organisations/ministry-of-defence', u'/ips-regional-passport-office', u'/hand-luggage-restrictions/overview', u'/jobseekers-allowance/eligibility', u'/register-to-vote', u'/disclosure-barring-service-check/overview', u'/browse/benefits/jobseekers-allowance', u'/dvlaforms', u'/tier-4-general-visa', u'/student-finance/loans-and-grants']

# function to retrieve GA data allowing filter on page name. None will retrieve overall GOVUK figures
# Returns dataframe which is result of average 10 Random Forest runs

def PageData(page, mod):
	'''
	args - pagepath for filter
		 - Model type - takes 'LR' for LASSO and 'RF' for Random Forests
	'''
	
	
	Store = []
	t = datetime.today()
	t2 = t - timedelta(hours=1)  #(2 hours for BST, 1 for UTC which is on Heroku server)
	delay = t2.strftime('%Y-%m-%d %H:00')
	star = t - timedelta(30)


	max_results = 5e7
	metrics = ['pageviews']
	dimensions = ['pagePath', 'hour', 'date']
	dim = ['date', 'hour']
	
	if page != None:
		filters = ['pagePath=='+page]
	else:
		filters = None

	df1 = ga.read_ga(metrics,
						 dimensions = dim,
						 start_date = star, 
						 end_date = delay, 
						 parse_dates=[['date', 'hour']],
						 token_file_name = 'static/token/analytics.dat',
						 secrets = 'static/token/client_secrets.json',
						 account_id = '26179049',
						 filters = filters
						 )

	##################### 48 MAX LAG ##############################

	ind = []

	for i in range(72, len(df1)):

		lag = [1,2,3,4,5,10,17,24,48,72]
		lagx = list(i - np.array(lag))
	
		Train = df1.ix[lagx]
		Target = df1.ix[i]

		TT = Train.T
		TT.columns = lag
		TT['Target'] = Target['pageviews']
		ind.append(TT)

	rng = date_range(df1.index[lag[-1]], df1.index[len(df1)-1], freq='H')
	Set = ind[0].append(ind[1:])
	Set.index = rng
	SetT = Set.ix[:delay]
	print SetT

	#############################################################

	li = []
	
	if mod == 'LR':
		cnt = 1
	else:
		cnt = 3
	
	feats = len(lag)
	SetT = SetT.replace(0,1)

	X_Train = SetT[SetT.columns[0:feats]][:-170]
	Y_Train = SetT['Target'].ix[:-170]


	X_Test = SetT[SetT.columns[0:feats]][-170:]
	Y_Test = SetT['Target'][-170:]
	
	Store.append(X_Train)
	Store.append(Y_Train)
	Store.append(X_Test)
	Store.append(Y_Test)
	
	
	for j in range(0,cnt):

		print j
	
		#Train Model
		# feats = len(lag)
# 		SetT = SetT.replace(0,1)
# 
# 		X_Train = SetT[SetT.columns[0:feats]][:-170]
# 		Y_Train = SetT['Target'].ix[:-170]
# 
# 
# 		X_Test = SetT[SetT.columns[0:feats]][-170:]
# 		Y_Test = SetT['Target'][-170:]

		
		if mod == 'RF':
		
			print 50*'-'
			print "Random Forest Regression"
			print 50*'-'
			rf = RandomForestRegressor(n_estimators=500, max_features=feats)
		
		else:
			print 50*'-'
			print "LASSO Regression"
			print 50*'-'

			################################################################################
			# Lasso with path and cross-validation using LassoCV path
			from sklearn.linear_model import LassoCV
# 
			lasso_cv = LassoCV()
# 
			y_ = lasso_cv.fit(X_Train, Y_Train)
			rf = y_ 
			
# 			rf = linear_model.LinearRegression()
		
		
		
		
		rf.fit(X_Train, Y_Train)
		PredRF = rf.predict(X_Test)
		scoreRF = r2_score(Y_Test,PredRF)
		MSE = np.mean(np.square(Y_Test-PredRF))
		print 'R2 Score = ', scoreRF 
		print 'MSE = ',  MSE

		Res = pd.DataFrame(columns=['res'], index = (range(0,len(Y_Test))))
		resid = Y_Test-PredRF
		Res['res'] = resid.values

		TSDU = Res['res'].mean()+3*Res['res'].std()
		TSDL = Res['res'].mean()-3*Res['res'].std()
		tsdP = Res[Res['res']>(Res['res'].mean()+3*Res['res'].std())]
		tsdN = Res[Res['res']<(Res['res'].mean()-3*Res['res'].std())]

		Stats = pd.DataFrame(columns=['yt','pred','resid','TSDU','TSDL'], index=X_Test.index)
		Stats['yt'] = Y_Test
		Stats['pred'] = PredRF
		Stats['resid'] = resid
		Stats['TSDU'] = TSDU
		Stats['TSDL'] = TSDL

		######### Plotting diabled for heroku build ############################

		# plt.figure(5)
		# 
		# plt.subplot(2, 1, 1)
		# Stats['yt'].plot()
		# plt.scatter(Stats['pred'].index, Stats['pred'], s=70, alpha=0.5, c='r')
		# 
		# plt.title("Random Forest Model")
		# 
		# plt.subplot(2,1,2)

		#######################################################################

		# Stats['resid'].plot()
		# Stats['TSDU'].plot(c='r')
		# Stats['TSDL'].plot(c='r')

		Stats.index.name = 'Time'

		Stats['time'] = Stats.index
		Stats['pred'].astype('int')
		Stats['resid'].astype('int')
		Stats['TSDU'].astype('int')
		Stats['TSDL'].astype('int')

		li.append(Stats)
	# plt.title('Residuals and 2 s.d. lines')

	cat = pd.concat(([i for i in li]))
	Stats = cat.groupby(cat.index).mean()
	Stats['time'] = Stats.index

	AP = len(tsdP) + len(tsdN)

	print 50*'-'  
	# print "PAGE: " + str(filters[0])
	print 50*'-'
	print "RANDOM FOREST Number of Anomalous Points is: %i" % AP
	print "%% Anomalous points is: %.1f" % (100*AP/len(Y_Test)) +'%'
	
	
	return Stats, Store, scoreRF


def arran(df, pag):
	'''
		arrange dataframe from pagedata function into suitable
		condensed format for matrix. values are % of 3.s.d marker
		therefore > value - > the closer to anomalous
	'''
	
	dfram = pd.DataFrame(columns=[pag], index=df.index)
	for i in range(0,len(df.index)):
		if df.ix[i,2] < 0:
			dfram.ix[i,0] = np.round(100*(df.ix[i,2]/df.ix[i,4]))
		else:
			dfram.ix[i,0] = np.round(100*(df.ix[i,2]/df.ix[i,3]))
	return dfram



def combine(num):
	'''
		take output from arran() and append values for each page. End result
		is a dataframe with index=page and column=time. The value of each cell 
		is the resid % of 3.s.d. threshold
		
		mod indicates model to use - LR for LASSO RF for Random Forest
	'''
	lzt = []
	lzt2 = []
	
	for i in range(0,num):
		#Random Forest
		f = PageData(top100[i], 'RF')
		R2RF = f[2]
		f = f[0]
		
		#Linear Reg
		ff = PageData(top100[i], 'LR')
		R2LR = ff[2]
		
		if R2RF > R2LR:
			print "Using Random Forest Model"
			f = f
		else:
			print "Using Linear Regression (LASSO) Model"
			f = ff[0]
		
		ftri = f.tail(25)   # just take last 24 hours
		g = arran(ftri,top100[i])
		lzt.append(g)
	fin = pd.concat([i.T for i in lzt], axis=0)	
	fin.index.name = 'Page'
	ss = fin.index.values
	ss[0] = 'GOVUK'
	fin.index = ss
		
	f3 = melt(fin)
	f3['page'] = list(fin.index.values)*len(fin.columns)
	f4 = f3.sort(['page','Time'])
	cls = ['page', 'Time', 'value']
	f4 = f4[cls]
	
	
	
	return f4


def combineMOD(start,end):
	'''
		take output from arran() and append values for each page. End result
		is a dataframe with index=page and column=time. The value of each cell 
		is the resid % of 3.s.d. threshold
		
		mod indicates model to use - LR for LASSO RF for Random Forest
	'''
	lzt =  []
	lzt2 = []
	R2R =  []
	R2L =  []
	
	for i in range(start,end):
		#Random Forest
		print 50*'*'
		print top100[i]
		print "Number %i of %i" %(i+1, end-start)
		print 50*'*'
		
		f = PageData(top100[i], 'RF')
		R2RF = f[2]
		f = f[0]
		
		#Linear Reg
		print 50*'*'
		print top100[i]
		print 50*'*'
		ff = PageData(top100[i], 'LR')
		R2LR = ff[2]
		ff = ff[0]
		

		
		ftri = f.tail(25)   # just take last 24 hours
		ftriLR = ff.tail(25)
		gRF = arran(ftri,top100[i])
		lzt.append(gRF)
		
		gLR = arran(ftriLR,top100[i])
		lzt2.append(gLR)
		
		R2R.append(R2RF)
		R2L.append(R2LR)
		
		print R2L
		
	fin = pd.concat([i.T for i in lzt], axis=0)	
	fin.index.name = 'Page'
	ss = fin.index.values

	if ss[0] == None:
		ss[0] = 'GOVUK'
	else:
		pass
		
	fin.index = ss
		
	f3 = melt(fin)
	f3['page'] = list(fin.index.values)*len(fin.columns)
	f4 = f3.sort(['page','Time'])
	cls = ['page', 'Time', 'value']
	f4 = f4[cls]
	
	finLR = pd.concat([i.T for i in lzt2], axis=0)	
	finLR.index.name = 'Page'
	ssLR = finLR.index.values

	if ssLR[0] == None:
		ssLR[0] = 'GOVUK'
 	else:
 		pass
	
	finLR.index = ssLR
		
	f3LR = melt(finLR)
	f3LR['page'] = list(finLR.index.values)*len(finLR.columns)
	f4LR = f3LR.sort(['page','Time'])
	cls = ['page', 'Time', 'value']
	f4LR = f4LR[cls]
	
	
	f4['valueLR'] = f4LR['value']
	
	f4.index = f4['page']
	t100L = top100
	t100L[0] = 'GOVUK'
	
	print f4
	
	f5 = f4.ix[t100L[start:end]]
		
	print f5
	
	f5['R2_RF'] = [R2R[i//25] for i in range(len(R2R)*25)]
	f5['R2_LR'] = [R2L[i//25] for i in range(len(R2L)*25)]
	
	f5['value_weight'] = (f5['value']*f5['R2_RF']+f5['valueLR']*f5['R2_LR'])/(f5['R2_RF']+f5['R2_LR'])
	
	return f5





# Stats.to_csv('static/data/PredTest.csv')

# factorize for heatmap

def fact(df):
	pge = pd.factorize(df['page'])
	time = pd.factorize(df['Time'])
	df['pageFCT'] = pge[0]
	df['TimeFCT'] = time[0]
	
	return df

# VV = combine(5, 'LR')  # this generates main data
# XX = fact(VV) # this factorises data from combine ready for heatmap

def neuro(page,neuron,goal):


	ss = PageData(page, 'LR')
	print ss[1][0]
	Xtrain = ss[1][0]
	Ytrain = ss[1][1]

	Xtest = ss[1][2]
	Ytest = ss[1][3]

	indim, odim = (Xtrain.shape)[1], (Ytrain.shape)

	targ = Ytrain.values.reshape(len(Ytrain),1)

	normXT = Xtrain.values/(np.max(Xtrain.values))
	normYT = Ytrain.values/(np.max(Ytrain.values))


	normXtst = Xtest.values/(np.max(Xtest.values))
	normYtst = Ytest.values/(np.max(Ytest.values))

	# Create network with 10 inputs, neuron neurons in input layer and 1 in output layer
	netx = nl.net.newff([[0,1]]*indim, [neuron,1])
	print "Layers: ", len(netx.layers)
	netx.trainf = nl.train.train_bfgs
	err = netx.train(normXT, normYT.reshape(len(normYT),1), epochs=10000, show=25, goal=goal)
	
	pred = []
	actual = []
	for j in range(0,len(normYtst)):
		pred.append(netx.sim([normXtst[j]])[0][0])
		actual.append(normYtst[j])

	df = pd.DataFrame(columns=['yt', 'pred', 'resid', 'TSDU', 'TSDL', 'time'], index=Xtest.index)
	df['yt'] = np.array(actual)*np.max(Ytest.values)
	df['pred'] = np.array(pred)*np.max(Ytest.values)
	df['resid'] = df['yt']-df['pred']		
	
	TSDU = df['resid'].mean()+3*df['resid'].std()
	TSDL = df['resid'].mean()-3*df['resid'].std()
	
	df['TSDU'] = TSDU
	df['TSDL'] = TSDL
	df['time'] = df.index

# 	df2 = df.ix[:-1]
	MSE = np.round(np.mean(np.square(df['resid'])))

	print "MSE: ", MSE

	return df, MSE

#  Dirty fix as net sometime fails to train if run in quick succession

def neu(page):   

	for i in range(0,10):
		DD = neuro(page,40,0.02)
		if DD[1]>np.square(np.max(DD[0]['resid'])):
			pass
		else:
			break
	return DD


################################## MYSQL TEST ##########################

def DBup():


	db = MySQLdb.connect(host='us-cdbr-east-06.cleardb.net',
		user='b82e44e89b8763',
		passwd='d2d0328b',
		db = 'heroku_87b1e1a28b1fa6b'
		)
	# 
	curs = db.cursor();
	# 
	# curs.execute("DROP TABLE IF EXISTS STATS2")
	curs.execute("DROP TABLE IF EXISTS STATS2")
	curs.execute("DROP TABLE IF EXISTS STATSLR")
	curs.execute("DROP TABLE IF EXISTS STATSNN")
	curs.execute("DROP TABLE IF EXISTS HEAT")
	curs.execute("DROP TABLE IF EXISTS TAB")
	curs.execute("DROP TABLE IF EXISTS DAYPLOT")


	# 
	# # sql2 = """CREATE TABLE TEST (
	# #          ID  INT NOT NULL,
	# #          COL1 CHAR(20),
	# #          COL2 CHAR(20),  
	# #          COL3 CHAR(20))"""
	# 
	# # curs.execute(sql2)
	# 
	print "Uploading Data To MySQL"
	# 
	# print Stats.tail()
	# 
	StatsRF.to_sql(con=db, name='STATS2', if_exists='replace', flavor='mysql')
	StatsLR.to_sql(con=db, name='STATSLR', if_exists='replace', flavor='mysql')
	StatsNN.to_sql(con=db, name='STATSNN', if_exists='replace', flavor='mysql')
	Heat2.to_sql(con=db, name='HEAT', if_exists='replace', flavor='mysql')
	TBL.to_sql(con=db, name='TAB', if_exists='replace', flavor='mysql')
	TimeSER.to_sql(con=db, name='DAYPLOT', if_exists='replace', flavor='mysql')

	
	sleep(2)

	print 50*'*'
	print "Database Check... "
	print 50*'*'

	print
	curs.execute("SELECT * FROM STATSNN")
	for i in curs:
		print i


############################################################################

###################### Data for upload to DB ###############################

# if __name__ == "__main__":
# 
# 	RF = []
# 	LR = []
# 	NN = []
# 	
# 	for i in range(0,10):
# 	
# 		Page = top100[i]
# 		SRF = PageData(Page, 'RF')
# 		SLR = PageData(Page, 'LR')
# 		SNN = neu(Page)
# 	
# 		StatsRF = SRF[0]  #Random Forest
# 		StatsLR = SLR[0]  #LASSO
# 		StatsNN = SNN[0]  #Neural Net
# 	
# 		RFMSE = np.mean(np.square(StatsRF['resid']))
# 		LRMSE = np.mean(np.square(StatsLR['resid']))
# 		NNMSE = np.mean(np.square(StatsNN['resid']))
# 	
# 		RF.append(RFMSE)
# 		LR.append(LRMSE)
# 		NN.append(NNMSE)
		
# 	Cont = combine(25)
# 	Heat2 = fact(Cont) # Heatmap


#Table view########################

def Tab(Cont):

	a = Cont[Cont['value_weight']>100]
	pge = list(a['page'].values)
	pg2p = [i+'_pred' for i in pge]

	for j in pg2p:
		pge.append(j)

	pge = list(set(pge))
	time = Cont['Time'][0:25]

# 	for k in range(0,len(pge)):
# 		if len(pge[k])>55:
# 			pge[k] = '/'.join(pge[k].split('/')[2:])

	ddf = pd.DataFrame(columns=['Page', 'Time', 'Actual', 'Pred'], index=range(0,a.shape[0]))
	DTS = pd.DataFrame(columns=pge, index=time)
	
	print DTS.columns
	
	DTS['Time'] = DTS.index

	ddf['Page'] = a['page'].values
	ddf['Time'] = a['Time'].values

	for i in range(0,len(ddf['Page'])):
		if ddf['Page'][i] == 'GOVUK':
			AR = PageData(None, 'RF')
			LR = PageData(None, 'LR')

			DTS['GOVUK'] = AR[0]['yt']

			if AR[2] > LR[2]:
				DTS['GOVUK_pred'] = AR[0]['pred']

			else:
				DTS['GOVUK_pred'] = LR[0]['pred']

		else:
			AR = PageData(ddf['Page'][i], 'RF')
			LR = PageData(ddf['Page'][i], 'LR')

			DTS[ddf['Page'][i]] = AR[0]['yt']

			if AR[2] > LR[2]:
				DTS[ddf['Page'][i]+"_pred"] = AR[0]['pred']
			else:
				DTS[ddf['Page'][i]+"_pred"] = LR[0]['pred']



		AR2 = AR[0].ix[ddf['Time'][i]]
		R2RF = AR[2]

		LR = PageData(ddf['Page'][i], 'LR')
		LR2 = LR[0].ix[ddf['Time'][i]]
		R2LR = LR[2]

		ddf['Actual'][i] = AR2['yt']
		ddf['Pred'][i] = (R2RF*AR2['pred'] + R2LR*LR2['pred'])/(R2LR+R2RF)

	return ddf, DTS
	
####################################

SRF = PageData(None, 'RF')
SLR = PageData(None, 'LR')
SNN = neu(None)
#  	
StatsRF = SRF[0]  #Random Forest
StatsLR = SLR[0]  #LASSO
StatsNN = SNN[0]  #Neural Net

Cont = combineMOD(0,50)
Heat2 = fact(Cont)
TTX = Tab(Cont)

TBL = TTX[0]
TimeSER = TTX[1]

pge = list(TimeSER.columns.values)

for k in range(0,len(pge)):
	if len(pge[k])>55:
		pge[k] = '/'.join(pge[k].split('/')[2:])

TimeSER.columns = pge


DBup()
