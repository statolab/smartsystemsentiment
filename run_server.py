from pickle import load
from keras.models import load_model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import statsmodels.api as sm
import itertools
from flask import Flask, render_template, session, redirect, url_for
from flask import send_file
import os
from flask import request
from flask_bootstrap import Bootstrap
from flask_wtf import Form
from wtforms import StringField, SubmitField, IntegerField, SelectField, Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange
### sentiment analysis libraries ###
import numpy as np
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from pickle import load
# from keras.models import load_model
# load the model
import pandas as pd
import re

### Plot Libraries ###
from flask import Flask, render_template,request
import plotly
import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import plotly.express as px

###########################################################
########################## flask ##########################
###########################################################

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
		
@app.errorhandler(404)
def page_not_found(e):
	return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
	return render_template('500.html'), 500

data = pd.read_csv('dataset.csv', encoding = "ISO-8859-1")

#top 20 words after removing stopwords
def get_top_n_words(corpus, n=None):
	vec = CountVectorizer(stop_words = 'english').fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]
common_words = get_top_n_words(data['SentimentText'], 20)
df2 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])

#top 20 bigrams after removing stopwords
def get_top_n_trigram(corpus, n=None):
	vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]
common_words = get_top_n_trigram(data['SentimentText'], 20)
df3 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])

@app.route('/', methods=['GET', 'POST'])
def init():
	return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('index.html')

@app.route('/sentiment-analysis', methods=['GET', 'POST'])
def index():
	df0 = []

	file_sample = 'dataset.csv'
	if request.method == 'POST':
		data = pd.read_csv(request.files.get('file'), encoding = "ISO-8859-1")
		sentimentcount = 20
	else:
		data = pd.read_csv(file_sample, encoding = "ISO-8859-1")
		sentimentcount = 8
	K.clear_session()
	model = load_model('sentiment_analysisEN10ep.h5')
	 
	# load the tokenizer
	tokenizer = load(open('sentiment_analysisEN10ep.pkl', 'rb'))

	print('---------------')
	output_review = []
	output_df = []
	show_table = min(len(data), sentimentcount)
	print([_ for _ in data['SentimentText'].values[0:10]])
	for i in range(show_table):
		review = [data['SentimentText'].values[i]]
		print(review)
		#vectorizing the tweet by the pre-fitted tokenizer instance
		review = tokenizer.texts_to_sequences(review)
		#padding the tweet to have exactly the same shape as `embedding_2` input
		review = pad_sequences(review, maxlen=712, dtype='int32', value=0)
		#print(review)
		sentiment = model.predict(review,batch_size=1,verbose = 2)[0]
		if(np.argmax(sentiment) == 0):
				print('negative')
				df0 += ['negative']
		elif (np.argmax(sentiment) == 1):
				print("positive")
				df0 += ['positive']
	neg_count = df0.count('negative')
	pos_count = df0.count('positive')

	text_limit = 200
	text_dataset = [_[:text_limit] + '...' if len(_)>text_limit else _ for _ in data['SentimentText'].values[0:show_table]]
	text_more = [_ if len(_)>text_limit else '' for _ in data['SentimentText'].values[0:show_table]]
	
	common_words = get_top_n_words(data['SentimentText'], 20)
	df2 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])
	common_words = get_top_n_trigram(data['SentimentText'], 20)
	df3 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])
	
	feature = 'Top 20 Words'
	bar = create_plot(feature)
	data1 = [go.Bar(x=df2['SentimentText'], y=df2['count'])]
	graphJSON = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)
	
	# df0.to_frame(name = 'hasil analisis')
	# df0.head(show_table)

	labels = ['Positive','Negative']
	values = [pos_count, neg_count]
	datapie =[go.Pie(labels=labels, values=values)] #pie chart jml sentimen +/-
	graphJSONpie = json.dumps((datapie), cls=plotly.utils.PlotlyJSONEncoder)
	
	return render_template(
		'index_sentimental_analysis.html',
		number =  range(len(df0)),
		output = df0,
		text_dataset = text_dataset,
		text_more = text_more,
		text_limit = text_limit,
		neg_count = neg_count,
		pos_count = pos_count, 
		graphJSON = graphJSON,
		graphJSONpie = graphJSONpie, 
		plot = bar,
	)

@app.route('/forecast', methods=['GET', 'POST'])
def indexfrc():
###############################################################
################### Grafik Data ###############################
###############################################################
	df = []
	filefrc_sample = 'datasetfrc.csv'

	if request.method == 'POST':
		df = pd.read_csv(request.files.get('file'), encoding = "ISO-8859-1")
		labeldata = df.columns.values[1]
		y = df[str(labeldata)]
	else:
		df = pd.read_csv(filefrc_sample, encoding = "ISO-8859-1")
		labeldata = df.columns.values[1]
		y = df[str(labeldata)]
	
###########################################################################################
################################## SARIMAX Forecast #######################################
###########################################################################################
		

	#df = pd.read_csv('/gdrive/My Drive/Proyek/Time Series Forecasting/sales report/total-business-sales_1.csv', usecols=[3])
	#df = pd.read_csv('/gdrive/My Drive/Proyek/Time Series Forecasting/monthly-beer-production-in-austr.csv', usecols=[1])
	#df = pd.read_csv('/gdrive/My Drive/Proyek/Time Series Forecasting/iottrend.csv', usecols=[1])
	
	#y.plot(figsize=(15,6))
	#df.plot.hist(figsize=(8,6))
	p = d = q = range(0, 2)
	pdq = list(itertools.product(p, d, q))
	seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

	dataparam = list()
	dataparamS = list()
	datares = list()
	datares2 = list()

	for param in pdq:
		for param_seasonal in seasonal_pdq:
				mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

				results = mod.fit() 
				dataparam.append(param)
				dataparamS.append(param_seasonal)
				datares.append(results.aic)
				datares2.append(results.aic)
				
	datares.sort()

	#################################################################################
	################### Output untuk tabel parameter SARIMAX ########################
	#################################################################################

	final_result = [[
		', '.join('%s' %dp for dp in dataparam[datares2.index(datares[_])]),
		', '.join('%s' %dp for dp in dataparamS[datares2.index(datares[_])]),
		datares[_]
		] for _ in range(3)]

	#p,d,q account for seasonality, trend, and noise
	mod = sm.tsa.statespace.SARIMAX(y,
									order=dataparam[datares2.index(datares[0])],
									seasonal_order=dataparamS[datares2.index(datares[0])],
									enforce_stationarity=False,
									enforce_invertibility=False)

	results = mod.fit()

	bestaic=datares[0]
	bestparam=dataparam[datares2.index(datares[0])]
	bestparamS=dataparamS[datares2.index(datares[0])]

	print(results.summary().tables[1])

	pred = results.get_prediction(dynamic=False)
	pred_ci = pred.conf_int()

	#################################################################################
	################### Jumlah Output untuk Forecast SARIMAX (steps) ################
	#################################################################################
	stepspercentage = 0.25
	dtcount = len(df[labeldata].index)

	pred_uc = results.get_forecast(steps=int(round(dtcount*stepspercentage)))
	steps=int(round(dtcount*stepspercentage))
	pred_ci = pred_uc.conf_int()
	fig = go.Figure()

	#plt.legend()
	#plt.subplot(2, 1, 1)

	y_forecasted = pred.predicted_mean
	y_truth = y[:]

	#######################################################
	################### Output MSE ########################
	#######################################################

	mse = ((y_forecasted - y_truth) ** 2).mean()


	countv2=y.count()
	maxv2=y.max()
	minv2=y.min()
	meanv2=y.mean()
	stdv2=y.std()


	dataplot=[go.Line(y=y[:])]
	graphJSONdata = json.dumps((dataplot), cls=plotly.utils.PlotlyJSONEncoder)

	featurefrc = 'SARIMAX'
    # print('data index : ', idxres)
    # print('with parameter : ', dataparam[datares2.index(datares[1])])
    # print('with seasonal parameter : ', dataparamS[datares2.index(datares[1])]

	if featurefrc == 'SARIMAX':
		startpercentage = 0.5
		A = len(df[labeldata].index)
		startpoint = round(dtcount*startpercentage)
		x1 = np.linspace(0, dtcount, dtcount)
		x2 = np.linspace(A-startpoint, A+steps, A+steps)
		dfx1 = pd.DataFrame({'x1':x1})
		dfx2 = pd.DataFrame({'x2':x2})
		datafrc=[go.Line(x=dfx1['x1'], y=y[startpoint:])]
		datafrc2=[go.Line(x=dfx2['x2'],y=pred_uc.predicted_mean)]
        
	else:
		N = 1000
		random_x = np.random.randn(N)
		random_y = np.random.randn(N)

        # Create a trace
		datafrc = [go.Scatter(x = random_x,y = random_y,mode = 'markers')]


	graphJSONfrc = json.dumps((datafrc + datafrc2), cls=plotly.utils.PlotlyJSONEncoder)
	output_sum = {
		'count' : countv2,
		'max' : maxv2,
		'min' : minv2,
		'mean' : meanv2,
		'std' : stdv2
	}
	return render_template('index_forecast.html', output_sum=output_sum, bestaic=bestaic,bestparam=bestparam,bestparamS=bestparamS,graphJSONfrc = graphJSONfrc, graphJSONdata = graphJSONdata, final_result = final_result)

	########## text frequency ##########
#top 20 words before removing stopwords
#data['Sentiment'] = data['Sentiment'].replace(0,'negative')
#data['Sentiment'] = data['Sentiment'].replace(1,'positive')
#data = data[data.Sentiment != "positive"]
	
def create_plot(feature):
	#top 20 words after removing stopwords
	if feature == 'Top 20 Words':
		#fig2 = px.bar(df2, x='SentimentText', y='count') #top words plot
		#fig2.show()
		data1 = []
		data1 = [go.Bar(x=df2['SentimentText'], y=df2['count'])]
		graphJSON = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)
	elif feature == 'Top 20 Bigrams' :
		#fig2 = px.bar(df3, x='SentimentText', y='count') #top bigrams plot
		#fig2.show()
		data2 = []
		data2 = [go.Bar(x=df3['SentimentText'], y=df3['count'])]
		graphJSON = json.dumps(data2, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():
	feature = request.args['selected']
	graphJSON = create_plot(feature)
	return graphJSON
	
@app.route('/getDatasetCSV')
def plot_csv():
	return send_file(
		'dataset.csv',
		mimetype='text/csv',
		attachment_filename='data set.csv',
		as_attachment=True
	)

#print('The Mean Squared Error of the forecasts is {}'.format(round(mse, 2)))
#print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#print('Data Count : ', countv.strip('Xiaomi    '))
#print('Maximum Value : ', countmx.strip('Xiaomi    '))
#print('Minimum Value : ', countmi.strip('Xiaomi    '))
#print('Mean : ', countme.strip('Xiaomi    '))
#print('Standard Deviation : ', countstd.strip('Xiaomi    '))
if __name__ == '__main__':
	app.run(host='0.0.0.0')
	# manager.run()