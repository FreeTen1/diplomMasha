import streamlit as st
import plotly 
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import json
from bs4 import BeautifulSoup
import requests
import datetime
from datetime import date,  timedelta
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from pmdarima.arima.utils import ndiffs
import jhtalib as jhta
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def Load_data():
#Парсинг с Yahoo Finance и создание датафрейма с данными

	#Парсинг перечня акций,с html-страницы
    dict_stocks = {}
    for i in range(0, 251, 250):
        html = requests.get(f'https://finance.yahoo.com/screener/predefined/ms_technology?count=250&offset={i}').content
        soup = BeautifulSoup(html, 'lxml')
        tbody = soup.find('tbody')
        stock =[]
        while not stock:
            stock = tbody.find_all('a')
        company_name = tbody.find_all('td', class_='Va(m) Ta(start) Px(10px) Fz(s)')
        dict_stocks.update({f"{stock[j].get_text()}": company_name[j].get_text() for j in range(len(stock))})

	#Парсинг ценовой истории для каждой акции за месяц, через API
    now = datetime.datetime.utcnow()
    dateto = int(now.timestamp()) 
    datefrom = int((now-timedelta(days=30)).timestamp()) 

    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?&period1={datefrom}&period2={dateto}&interval=1d&includePrePost=true")
    dates_timestamp = json.loads(r.content)['chart']['result'][0]['timestamp']
    dates = []	#столбцы для датафрейма - торговые даты
    for item in dates_timestamp:
        dates.append(date.fromtimestamp(item))
    dates.insert(0,'Name')
    dates.insert(1,'CompanyName')

    counts=0
    placeholder=st.empty()
    prices_list =[]   
    with st.empty():
	    for stock, company_name in dict_stocks.items():
	        try: 
	            counts+=1
	            st.write(f"⏳ {round(counts/len(dict_stocks)*100)} % данных загружено") 

	            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{stock}?&period1={datefrom}&period2={dateto}&interval=1d&includePrePost=true")
	            temp_prices_list = json.loads(r.content)['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
	            temp_prices_list.insert(0,stock) 
	            temp_prices_list.insert(1,company_name)
	            prices_list.append(temp_prices_list) 

	            placeholder.empty()
	        except:	     		            
	            continue 

	#Создание общего датафрейма
    df_prices = pd.DataFrame(prices_list, columns=dates).dropna()
    df_prices.reset_index(inplace=True)
    df_prices.pop('index')
    return df_prices

@st.cache(suppress_st_warning=True)   
def Clustering(df):
#Кластеризация, метод DBSCAN

	df_cl = df.copy()
	X = df_cl.iloc[:,2:].to_numpy()

	#Количество кластеров по макс. силуэту
	scores = []
	values = np.arange(1, 1000)
	for temp_eps in values:
		try:
			dbscan = DBSCAN(eps=temp_eps/1000, min_samples=2, metric='correlation').fit(X)
			score = silhouette_score(X, dbscan.labels_, metric='correlation', sample_size=len(X))
			scores.append(score)  
		except:
			scores.append(0)
			continue  
	optimal_eps = np.argmax(scores)+1

	#Кластеризация
	dbscan = DBSCAN(eps=optimal_eps/1000, min_samples=2, metric='correlation').fit(X)
	df_cl['num_clust'] = dbscan.labels_+1
	silhouette = round(max(scores)*100)

	return df_cl, silhouette


@st.cache(suppress_st_warning=True)   
def Predict(Name):
#Прогнозирование, модель ARIMA

	#Загрузка исторических данных
	df_3years = pd.DataFrame(pdr.get_data_yahoo(Name,
		start=date.today()-timedelta(days=365*3),
		end=date.today())['Adj Close'])
	
	#Подбор параметров для модели
	best_model = auto_arima(
		df_3years,
		max_q=20,
		max_p=20, 
		d=ndiffs(df_3years,test='adf'),
		trend='t',
		trace=True,
		stepwise=True)

	#Инициализация и обучение модели
	model = sm.tsa.statespace.SARIMAX(
		df_3years,
		order=best_model.order,
		trend='t',
		stepwise=True)
	result = model.fit()

	#Предсказание на 10 шагов
	prediction_frame = result.get_forecast(10).summary_frame()
	prediction = list(prediction_frame['mean'])	#среднее значение	
	prediction.insert(0,df_3years['Adj Close'][-1])
	prediction_min = prediction_frame.mean_ci_lower	#доверительный интервал
	prediction_max = prediction_frame.mean_ci_upper

	#Даты - будние дни
	prediction_days = []
	prediction_days.append(df_3years.index[-1])
	delta = 1
	while len(prediction_days)<10:
		day = datetime.datetime.today()+timedelta(days=delta)
		if day.weekday()<5:
			prediction_days.append(day)
		delta+=1

	return prediction, prediction_max, prediction_min, prediction_days

@st.cache(suppress_st_warning=True)
def Avg_levels(prices):
#Вычисление 2-х скользящих средних и точек их пересечения, уровней сопротивления и поддержки

	#Вычисление длинной и короткой скользящих средних
	short_avg = prices.rolling(window=5).mean()
	long_avg = prices.rolling(window=15).mean()

	#Точки пересечения скользящих средних
	prices['short']=short_avg[-len(prices):]
	prices['long']=long_avg[-len(prices):]

	previous_short = prices['short'].shift(1)
	previous_long = prices['long'].shift(1)

	sell = (prices['short'] <= prices['long']) & (previous_short >= previous_long)
	dates1 = sell[sell==True].index
	sell_df = prices.shift(1).loc[dates1]

	buy = (prices['short'] >= prices['long']) & (previous_short <= previous_long)
	dates2 = buy[buy==True].index
	buy_df = prices.shift(1).loc[dates2]

	#Вычисление уровней:
	pr = prices.values[:,0]

	#Минимальный из наиболее часто встречающихся локальных минимумов
	pr_min = []
	for i in range(len(pr)):	
	    if  pr[i] < min(pr[i - 1], pr[(i + 1) % len(pr)]):
	        pr_min.append(round(pr[i]))
	pr_min.sort()
	pr_min = pr_min[:round(len(pr_min)/2)]
	y_low = min(a for a in pr_min if pr_min.count(a) == max(map(pr_min.count, pr_min)))	

	#Максимальный из наиболее часто встречающихся локальных максимумов
	pr_max = []
	for i in range(len(pr)):	
	    if  pr[i] > max(pr[i - 1], pr[(i + 1) % len(pr)]):
	        pr_max.append(round(pr[i]))
	pr_max.sort(reverse=True)
	pr_max = pr_max[:round(len(pr_max)/2)]
	y_high = max(a for a in pr_max if pr_max.count(a) == max(map(pr_max.count, pr_max)))	

	return short_avg, long_avg, sell_df, buy_df, y_high, y_low

@st.cache(suppress_st_warning=True)
def Sell_buy(df):
#Выбор акций для продажи и покупки через полосы Боллинджера 
#Продажа: уровень цены над верхней полосой и падение цены за последние сутки >=10%
#Покупка: уровень цены под нижней полосой, и прирост цены за последние сутки >=10%

	sell_list = []
	buy_list = []

	for item in df['Name']:

		prices = df.loc[df['Name']==item].T.iloc[2:]
		price = pd.DataFrame()	
		price['Adj Close'] = prices.values[:,0]
		price['Close'] = prices.values[:,0]
		price['High'] = prices.values[:,0]
		price['Low'] = prices.values[:,0]

		bbands_levels = jhta.BBANDS(price, 14)	#вычисление полос Боллинджера

		dif = prices.values[-1,:]/prices.values[-2,:]	#последнее изменение цены

		if prices.values[-2,:]<=bbands_levels['lowerband'][-1] and dif>=1.1:
			dif_percent = round((dif[0]-1)*100)
			buy_temp_list = [item,
				bbands_levels['lowerband'][13:],
				bbands_levels['upperband'][13:],
				bbands_levels['midband'][13:],
				dif_percent]
			buy_list.append(buy_temp_list)

		if prices.values[-2,:]>=bbands_levels['upperband'][-1] and dif<=0.9:
			dif_percent = round(abs(1-dif[0])*100)
			sell_temp_list = [item,
				bbands_levels['lowerband'][13:],
				bbands_levels['upperband'][13:],
				bbands_levels['midband'][13:],
				dif_percent]
			sell_list.append(sell_temp_list)
			
	buy_df = pd.DataFrame(buy_list,columns=['Name','Lower','Upper','Mid','Dif'])
	sell_df = pd.DataFrame(sell_list,columns=['Name','Lower','Upper','Mid','Dif'])

	return buy_df, sell_df

@st.cache(suppress_st_warning=True)
def WordEnd(x):
#Окончание в зависимости от числа
    f1 = lambda x: (x%100)//10 != 1 and x%10 == 1
    f2 = lambda x: (x%100)//10 != 1 and x%10 in [2,3,4]
    return "акция" if f1(x) else "акции" if f2(x) else "акций"

#=============================================================

st.set_page_config(page_title="Stocks Analysis", page_icon=":bar_chart:", initial_sidebar_state="expanded",layout="wide")
selected_function =  st.sidebar.radio('', ['Схожая динамика','Продажа и покупка'], False)

#Загрузка данных
df = Load_data()

#1я радиокнопка, кластеризация
if selected_function == 'Схожая динамика':

	df_cl, sil = Clustering(df)
	st.write(f'Точность {sil} %')

	#Визуализация результатов
	col1,col2,col3 = st.beta_columns([1,0.5,1])
	col4,col5,col6 = st.beta_columns([1,0.5,1])

	df_col = df_cl.groupby('num_clust').size().sort_values(0,ascending=False)

	for j in range(len(df_col)):

		fig = plt.figure()
		fig.patch.set_facecolor("#0E1117")

		ax = fig.add_subplot(111, projection='3d')
		ax.patch.set_facecolor("#0E1117")	
		ax.xaxis.pane.fill = False
		ax.yaxis.pane.fill = False
		ax.zaxis.pane.fill = False
		ax.grid(False)
		ax.view_init(20,-80)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])

		for i in df_cl.loc[df_cl['num_clust']==df_col.index[j]].sort_values(df.columns[-1]).index[:3]:
			Axes3D.plot(ax, range(df_cl.shape[1]-3), [0]*(df_cl.shape[1]-3), df_cl.iloc[i,2:-1])

		kol = df_cl.groupby('num_clust').size().values[df_col.index[j]]

		word = WordEnd(kol)
		names = df_cl.loc[df_cl['num_clust']==df_col.index[j]].CompanyName.sort_values()
		if j%2==0:
			col1.write(f'**{kol} {word}:**')
			col1.write(fig)
			with col1.beta_expander('Список'):
				for index in names.index:
					st.write(df_cl.CompanyName[index])
			col1.write('')
		else:
			col3.write(f'**{kol} {word}:**')
			col3.write(fig)
			with col3.beta_expander('Список'):
				for index in names.index:
					st.write(df_cl.CompanyName[index])
			col3.write('')
		
#2я радиокнопка
if selected_function == 'Продажа и покупка':
	
	buy_df, sell_df = Sell_buy(df)

	with st.beta_expander('Продажа',): 
		if sell_df.empty:
			st.write('На сегодня предложений нет')
		else:
			for item_name in sell_df['Name']:
				company_name = df.loc[df['Name']==item_name].CompanyName.values[0]
				st.write(f'**{item_name}**: {company_name}')
				dif = buy_df.loc[buy_df['Name']==item].Dif.values[0]
				st.write(f'Просадка: {dif} % :small_red_triangle_down:')
				prices = df.loc[df['Name']==item_name].T.iloc[2:]
				pd.to_datetime(prices.index)

				fig = go.Figure()	
				fig.add_trace(go.Scatter(
					x=prices.index, 
					y=prices.values[:,0],
					line=dict(color='#660099', width=4),
					name='Стоимость'))

				fig.add_trace(go.Scatter(
					x=prices.index[-8:], 
					y=sell_df.loc[buy_df['Name']==item].Mid.values[0],
					line=dict(color='black', width=1),
					mode='lines',
					name='14MSA'))

				fig.add_trace(go.Scatter(
					x=prices.index[-8:], 
					y=sell_df.loc[buy_df['Name']==item].Upper.values[0],
					fill=None,
					line=dict(color='red', width=2),
				    mode='lines',
					name='70%'))

				fig.add_trace(go.Scatter(
					x=prices.index[-8:], 
					y=sell_df.loc[buy_df['Name']==item].Lower.values[0],
			    	fill='tonexty',
			    	line=dict(color='green', width=2), 
			    	mode='lines',
			    	name='30%'))

				fig.update_layout(
					{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932", 'legend_orientation': "h"},
					margin=dict(l=10, r=5, t=60, b=10),
					legend=dict(x=.5, xanchor="center"),
					title=f"Полосы Боллинджера",
					yaxis_title="$", 
					font=dict(size=15,color='white'), 
					dragmode='zoom',hovermode='x unified',
					autosize=False, width=1000, height=300)	
					
				st.write(fig)

	with st.beta_expander('Покупка',): 
		if buy_df.empty:
			st.write('На сегодня предложений нет')
		else:
			for item in buy_df['Name']:
				company_name = df.loc[df['Name']==item].CompanyName.values[0]
				st.write(f'**{item}**: {company_name}')
				dif = buy_df.loc[buy_df['Name']==item].Dif.values[0]
				st.write(f'Прирост: {dif} % :small_red_triangle:')
				prices = df.loc[df['Name']==item].T.iloc[2:]
				pd.to_datetime(prices.index)

				fig = go.Figure()
				fig.add_trace(go.Scatter(
					x=prices.index, 
					y=prices.values[:,0],
					line=dict(color='#660099', width=4),
					name='Стоимость'))

				fig.add_trace(go.Scatter(
					x=prices.index[-7:], 
					y=buy_df.loc[buy_df['Name']==item].Mid.values[0],
					line=dict(color='black', width=1),
					mode='lines',
					name='14MSA'))

				fig.add_trace(go.Scatter(
					x=prices.index[-7:], 
					y=buy_df.loc[buy_df['Name']==item].Upper.values[0],
					fill=None,
					line=dict(color='red', width=2),
				    mode='lines',
					name='70%'))

				fig.add_trace(go.Scatter(
					x=prices.index[-7:], 
					y=buy_df.loc[buy_df['Name']==item].Lower.values[0],
			    	fill='tonexty',
			    	line=dict(color='green', width=2), 
			    	mode='lines',
			    	name='30%'))

				fig.update_layout(
					{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932", 'legend_orientation': "h"},
					margin=dict(l=10, r=5, t=60, b=10),
					legend=dict(x=.5, xanchor="center"),
					title=f"Полосы Боллинджера",
					yaxis_title="$", 
					font=dict(size=15,color='white'), 
					dragmode='zoom',hovermode='x unified',
					autosize=False, width=1000, height=300)	

				st.write(fig)

#Выбор отдельных акций
selected_stocks = st.sidebar.multiselect('Выберите акции:', df['CompanyName'])
if selected_stocks:

	graf = st.sidebar
	graf.header('Параметры:')
	check_price = graf.checkbox('Стоимость',True)
	check_earn = graf.checkbox('Доходность')
	check_statistic = graf.checkbox('Уровни и средние')	
	check_forecast = graf.checkbox('Прогноз')

	for company_name in selected_stocks:
		item_name = df.loc[df['CompanyName']==company_name].Name.values[0]
		st.write(f'**{item_name}**: {company_name}')
		prices = df.loc[df['Name']==item_name].T.iloc[2:]
		pd.to_datetime(prices.index)

		#Цена
		if check_price:

			fig = go.Figure()
			fig.add_trace(go.Scatter(
				x=prices.index, 
				y=prices.values[:,0],
				line=dict(color='#660099', width=4),
				name='Стоимость'))

			fig.update_layout(
				{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932", 'legend_orientation': "h"},
				margin=dict(l=10, r=5, t=60, b=10),
				legend=dict(x=.5, xanchor="center"),
				title=f"Стоимость",
				yaxis_title="$", 
				font=dict(size=15,color='white'), 
				dragmode='zoom',hovermode='x unified',
				autosize=False, width=1000, height=300
				)	
				
			st.write(fig)

		#Доходность
		if check_earn:

			daily_returns = prices.diff(1)
			daily_returns.fillna(0, inplace=True)
			plus_earn = daily_returns[daily_returns>0]
			minus_earn = daily_returns[daily_returns<0]

			fig = go.Figure()
			fig.add_trace(go.Bar(
				x=plus_earn.index, 
				y=plus_earn.values[:,0],
				name='Доход'))

			fig.add_trace(go.Bar(
				x=minus_earn.index, 
				y=minus_earn.values[:,0],
				name='Убыток'))

			fig.update_layout(
				{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932"},
				margin=dict(l=10, r=5, t=60, b=10),
				title=f"Дневная доходность",
			    yaxis_title="$", 
			    font=dict(size=15, color='white'), 
			    dragmode='zoom', hovermode='y unified',
			    autosize=False, width=1000, height=300,
				yaxis=dict(showgrid=True,automargin=True),
				barmode='stack', showlegend=False)
			st.write(fig)

		#Средние и уровни 
		if check_statistic:

			short_moving_avg, long_moving_avg, sell_df, buy_df, y_high, y_low = Avg_levels(prices)

			fig = go.Figure()
			fig.add_trace(go.Scatter(
				x=prices.index, 
				y=prices.values[:,0],
				line=dict(color='#660099', width=4),
				name='Стоимость'))

			fig.add_trace(go.Scatter(
				x=short_moving_avg.index,
				y=short_moving_avg.values[:,0],
				name = f'5SMA', 
				mode='lines',
				line=dict(color='orange', width=2)))

			fig.add_trace(go.Scatter(
				x=long_moving_avg.index,
				y=long_moving_avg.values[:,0],
				name = f'15SMA', 
				mode='lines',
				line=dict(color='green', width=2)))

			line_low = [y_low]*len(prices)
			line_high = [y_high]*len(prices)

			fig.add_trace(go.Scatter(
				x=sell_df.index, 
				y=sell_df['long'], 
				mode='markers',
				marker=dict(
		        color='Red',
		        symbol=6,
		        size=20),
		        name='Продавать'))

			fig.add_trace(go.Scatter(
				x=buy_df.index,
				y=buy_df['long'], 
				mode='markers',
				marker=dict(
		        color='Green',
		        symbol=5,
		        size=20),
				name='Покупать'))

			fig.add_trace(go.Scatter(
				x=prices.index,
				y=line_low,
				name = 'Уровень поддержки', 
				line=dict(color='blue', width=2)))

			fig.add_trace(go.Scatter(
				x=prices.index,
				y=line_high,
				name = 'Уровень сопротивления', 
				line=dict(color='red', width=2)))

			fig.update_layout(
				{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932", 'legend_orientation': "h"},
				margin=dict(l=10, r=5, t=60, b=10, pad=0),
				legend=dict(x=.6, xanchor="center"),
				title=f"Уровни и средние",
				yaxis_title="$", 
				font=dict(size=15,color='white'), 
				dragmode='zoom',hovermode='x unified',
				autosize=False, width=1000, height=400)	

			st.write(fig)

		#Прогноз
		if check_forecast:

			prediction, prediction_max, prediction_min, prediction_dates = Predict(item_name)

			fig = go.Figure()
			fig.add_trace(go.Scatter(
				x=prediction_dates, 
				y=prediction,
				line=dict(color='red', width=4),
				mode='lines',
				name='Прогноз'))

			fig.add_trace(go.Scatter(
				x=prices.index, 
				y=prices.values[:,0],
				line=dict(color='#660099', width=4),
				name='Стоимость'))

			fig.update_layout(
				{'plot_bgcolor': "#D6DBDF", 'paper_bgcolor': "#272932", 'legend_orientation': "h"},
				margin=dict(l=10, r=5, t=60, b=10),
				legend=dict(x=.5, xanchor="center"),
				title=f"Прогноз",
				yaxis_title="$", 
				font=dict(size=15,color='white'), 
				dragmode='zoom',hovermode='x unified',
				autosize=False, width=1000, height=300)	
				
			st.write(fig)
					



