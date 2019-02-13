
"""****************************
Pairs Trading Strategy
****************************"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
from zipline.utils import tradingcalendar
import pytz


def initialize(context):
    # Quantopian backtester specific variables
    set_symbol_lookup_date('2014-01-01')

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))                  
            
    context.stock_pairs = [(symbol('ABGB'), symbol('FSLR')),
                           (symbol('CSUN'), symbol('ASTI')),
                           (symbol('KO'),   symbol('PEP')),
                           (symbol('AAPL'), symbol('IBM')),
                           (symbol('FB'),   symbol('YHOO')),
                           (symbol('TWTR'), symbol('YHOO'))]    
    
    context.stocks = [symbol('ABGB'), symbol('FSLR'), symbol('CSUN'), symbol('ASTI'),\
                      symbol('KO'), symbol('PEP'), symbol('AAPL'), symbol('IBM'), symbol('FB'),\
                      symbol('YHOO'),symbol('TWTR')]
    
    context.num_pairs = len(context.stock_pairs)
    # strategy specific variables
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback
    
    context.spread = np.ndarray((context.num_pairs, 0))
    # context.hedgeRatioTS = np.ndarray((context.num_pairs, 0))
    context.inLong = [False] * context.num_pairs
    context.inShort = [False] * context.num_pairs
    
    schedule_function(func=check_pair_status, date_rule=date_rules.every_day(),\
                      time_rule=time_rules.market_close(minutes=90))   

def check_pair_status(context, data):
    if get_open_orders():
        return    

    prices = data.history(context.stocks, fields='price', bar_count=35, frequency='1d').\
             iloc[-context.lookback::]
    
    new_spreads = np.ndarray((context.num_pairs, 1))
    
    for i in range(context.num_pairs):

        (stock_y, stock_x) = context.stock_pairs[i]

        Y = prices[stock_y]
        X = prices[stock_x]

        try:
            hedge = hedge_ratio(Y, X, add_const=True)   
            record(hedge=hedge)
        except ValueError as e:
            log.debug(e)
            return

        # context.hedgeRatioTS = np.append(context.hedgeRatioTS, hedge)
        
        new_spreads[i, :] = Y[-1] - hedge * X[-1]

        if context.spread.shape[1] > context.z_window:
            # Keep only the z-score lookback period
            spreads = context.spread[i, -context.z_window:]

            zscore = (spreads[-1] - spreads.mean()) / spreads.std()
            record(zscore=zscore)
            
            if context.inShort[i] and zscore < 0.0:
                order_target(stock_y, 0)
                order_target(stock_x, 0)
                context.inShort[i] = False
                context.inLong[i] = False
                record(X_pct=0, Y_pct=0)
                return

            if context.inLong[i] and zscore > 0.0:
                order_target(stock_y, 0)
                order_target(stock_x, 0)
                context.inShort[i] = False
                context.inLong[i] = False
                record(X_pct=0, Y_pct=0)
                return

            if zscore < -1.0 and (not context.inLong[i]):
                # Only trade if NOT already in a trade
                y_target_shares = 1       #long y
                X_target_shares = -hedge  #short x
                context.inLong[i] = True
                context.inShort[i] = False

                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares,X_target_shares, Y[-1], X[-1] )
                order_target_percent( stock_y, y_target_pct * (1.0/context.num_pairs) )
                order_target_percent( stock_x, x_target_pct * (1.0/context.num_pairs) )
                record(Y_pct=y_target_pct, X_pct=x_target_pct)
                return

            if zscore > 1.0 and (not context.inShort[i]):
                # Only trade if NOT already in a trade
                y_target_shares = -1     #short y
                X_target_shares = hedge  #long x
                context.inShort[i] = True
                context.inLong[i] = False

                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, Y[-1], X[-1] )
                order_target_percent( stock_y, y_target_pct * (1.0/context.num_pairs))
                order_target_percent( stock_x, x_target_pct * (1.0/context.num_pairs))
                record(Y_pct=y_target_pct, X_pct=x_target_pct)
        
    context.spread = np.hstack([context.spread, new_spreads])

def hedge_ratio(Y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model.params[1]
    model = sm.OLS(Y, X).fit()
    return model.params.values
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)

def handle_data(context, data):
    pass
	
	
	
	
	
"""****************************
Long Short Strategy
****************************"""

import numpy as np
import pandas as pd

from zipline.api import attach_pipeline, pipeline_output
import zipline.pipeline.pipeline as Pipeline #from quantopian.pipeline import Pipeline
from zipline.pipeline.factors.basic  import CustomFactor, SimpleMovingAverage, AverageDollarVolume
from zipline.pipeline.data import USEquityPricing
#from quantopian.pipeline.data import morningstar

import zipline.pipeline.filters.filter#from quantopian.pipeline.filters import Q1500US

# Constraint Parameters
NUM_LONG_POSITIONS = 5
NUM_SHORT_POSITIONS = 5

class Momentum(CustomFactor):

    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, prices):
        out[:] = ((prices[-21] - prices[-252])/prices[-252] -
                  (prices[-1] - prices[-21])/prices[-21])

def make_pipeline():
    
    # define alpha factors
    momentum = Momentum()
    growth = morningstar.operation_ratios.revenue_growth.latest
    pe_ratio = morningstar.valuation_ratios.pe_ratio.latest
        
    # Screen out non-desirable securities by defining our universe. 
    mkt_cap_filter = morningstar.valuation.market_cap.latest >= 500000000    
    price_filter = USEquityPricing.close.latest >= 5
    universe = Q1500US() & price_filter & mkt_cap_filter & \
               momentum.notnull() & growth.notnull() & pe_ratio.notnull()

    combined_rank = (
        momentum.rank(mask=universe).zscore() +
        growth.rank(mask=universe).zscore() +
        pe_ratio.rank(mask=universe).zscore()
    )

    longs = combined_rank.top(NUM_LONG_POSITIONS)
    shorts = combined_rank.bottom(NUM_SHORT_POSITIONS)

    long_short_screen = (longs | shorts)        

    # Create pipeline
    pipe = Pipeline(columns = {
        'longs':longs,
        'shorts':shorts,
        'combined_rank':combined_rank,
        'momentum':momentum,
        'growth':growth,            
        'pe_ratio':pe_ratio
    },
    screen = long_short_screen)
    return pipe

def initialize(context):

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))        

    attach_pipeline(make_pipeline(), 'long_short_factors')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(hours=1,minutes=30),
                      half_days=True)
    
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)

def before_trading_start(context, data):
    # Call pipeline_output to get the output
    context.output = pipeline_output('long_short_factors')
    
    context.longs = context.output[context.output['longs']].index.tolist()
    context.shorts = context.output[context.output['shorts']].index.tolist()

    context.long_weight, context.short_weight = assign_weights(context)
    
    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
   

def assign_weights(context):
    """
    Assign weights to securities that we want to order.
    """
    long_weight = 0.5 / len(context.longs)
    short_weight = -0.5 / len(context.shorts)
        
    return long_weight, short_weight
 
def rebalance(context, data):
    
    for security in context.portfolio.positions:
        if security not in context.longs and \
        security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)

    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, context.long_weight)

    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, context.short_weight)        
    
def recording_statements(context, data):
    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    # Record our variables.
    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)
	

	
	
	
	
"""****************************
Stochastic Volatility
****************************"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_datareader import data

import pymc3 as pm

np.random.seed(0)

def main():

    #load data    
    returns = data.get_data_google('SPY', start='2008-5-1', end='2009-12-1')['Close'].pct_change()
    returns.plot()
    plt.ylabel('daily returns in %');
    
    with pm.Model() as sp500_model:
        
        nu = pm.Exponential('nu', 1./10, testval=5.0)
        sigma = pm.Exponential('sigma', 1./0.02, testval=0.1)
        
        s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))                
        r = pm.StudentT('r', nu, lam=pm.math.exp(-2*s), observed=returns)
        
    
    with sp500_model:
        trace = pm.sample(2000)

    pm.traceplot(trace, [nu, sigma]);
    plt.show()
    
    plt.figure()
    returns.plot()
    plt.plot(returns.index, np.exp(trace['s',::5].T), 'r', alpha=.03)
    plt.legend(['S&P500', 'stochastic volatility process'])
    plt.show()

    
if __name__ == "__main__":
    main()
	
	
	
	


	
	
	
	
"""****************************
Recurrent Neural Network
****************************"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pandas_datareader import data
from datetime import datetime
import pytz

import matplotlib.pyplot as plt

np.random.seed(0)

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),0])
        dataY.append(dataset[i+look_back,0])

    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":

    #load data
    start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
    spy = data.DataReader("SPY", "google", start, end)
    dataset = np.array(spy['Close'].values).reshape(-1,1)
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape for look_back
    look_back = 10
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # LSTM
    model = Sequential()
    model.add(LSTM(32, input_dim=1)) #look_back))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=100, batch_size=5, verbose=2)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) 
   
    # scale back 
    train_pred = scaler.inverse_transform(train_pred)
    y_train = scaler.inverse_transform(y_train)
    test_pred = scaler.inverse_transform(test_pred)
    y_test = scaler.inverse_transform(y_test)
   
    # shift predictions for plotting
    train_pred_plot = np.empty_like(dataset)
    train_pred_plot[:,:] = np.nan
    train_pred_plot[look_back:len(train_pred)+look_back,:] = train_pred

    test_pred_plot = np.empty_like(dataset)
    test_pred_plot[:,:] = np.nan
    test_pred_plot[len(train_pred)+(look_back*2)+1:len(dataset)-1,:] = test_pred

    f = plt.figure()
    plt.plot(scaler.inverse_transform(dataset), color='b', lw=2.0, label='S&P 500')
    plt.plot(train_pred_plot, color='g', lw=2.0, label='LSTM train')
    plt.plot(test_pred_plot, color='r', lw=2.0, label='LSTM test')
    plt.legend(loc=3)
    plt.grid(True)
    f.savefig('./lstm.png')
	
	

	
"""****************************
Stock Clusters
****************************"""

import numpy as np
import pandas as pd
from scipy import linalg

from datetime import datetime
import pytz

from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, manifold

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

np.random.seed(0)

if __name__ == "__main__":
    
    num_samples = 60
    num_features = 20
    
    #generate data (synthetic)
    #prec = make_sparse_spd_matrix(num_features, alpha=0.95, smallest_coef=0.4, largest_coef=0.7)
    #cov = linalg.inv(prec)
    #X = np.random.multivariate_normal(np.zeros(num_features), cov, size=num_samples)
    #X = StandardScaler().fit_transform(X)    
   
    #generate data (actual)
    STOCKS = {
        'SPY': 'S&P500',
        'LQD': 'Bond_Corp',
        'TIP': 'Bond_Treas',
        'GLD': 'Gold',
        'MSFT': 'Microsoft',
        'XOM':  'Exxon',
        'AMZN': 'Amazon',
        'BAC':  'BofA',
        'NVS':  'Novartis'}
      
    symbols, names = np.array(list(STOCKS.items())).T
   
    start = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)    

    quotes = [quotes_historical_yahoo(symbol, start, end, asobject=True) for symbol in symbols]

    qopen = np.array([q.open for q in quotes]).astype(np.float)
    qclose = np.array([q.close for q in quotes]).astype(np.float)                
            
    variation= qclose - qopen  #per day variation in price for each symbol
    X = variation.T
    X /= X.std(axis=0) #standardize to use correlations rather than covariance
                
    #estimate inverse covariance    
    graph = GraphLassoCV()
    graph.fit(X)
    
    gl_cov = graph.covariance_
    gl_prec = graph.precision_
    gl_alphas =graph.cv_alphas_
    gl_scores = np.mean(graph.grid_scores, axis=1)

    plt.figure()        
    sns.heatmap(gl_prec)
    
    plt.figure()    
    plt.plot(gl_alphas, gl_scores, marker='o', color='b', lw=2.0, label='GraphLassoCV')
    plt.title("Graph Lasso Alpha Selection")
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.legend()
    
    #cluster using affinity propagation
    _, labels = cluster.affinity_propagation(gl_cov)
    num_labels = np.max(labels)
    
    for i in range(num_labels+1):
        print("Cluster %i: %s" %((i+1), ', '.join(names[labels==i])))
    
    #find a low dim embedding for visualization
    node_model = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=6, eigen_solver='dense')
    embedding = node_model.fit_transform(X.T).T
    
    #generate plots
    plt.figure()
    plt.clf()
    ax = plt.axes([0.,0.,1.,1.])
    plt.axis('off')
    
    partial_corr = gl_prec
    d = 1 / np.sqrt(np.diag(partial_corr))    
    non_zero = (np.abs(np.triu(partial_corr, k=1)) > 0.02)  #connectivity matrix
    
    #plot the nodes
    plt.scatter(embedding[0], embedding[1], s = 100*d**2, c = labels, cmap = plt.cm.spectral)
    
    #plot the edges
    start_idx, end_idx = np.where(non_zero)
    segments = [[embedding[:,start], embedding[:,stop]] for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_corr[non_zero])
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0,0.7*values.max()))
    lc.set_array(values)
    lc.set_linewidths(5*values)
    ax.add_collection(lc)
    
    #plot the labels
    for index, (name, label, (x,y)) in enumerate(zip(names, labels, embedding.T)):
        plt.text(x,y,name,size=12)
		
		
		
		
		

"""****************************
Gaussian Process Regression
****************************"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data, wb

from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split

from datetime import datetime
import pytz

np.random.seed(0)

def f(x): return x * np.sin(x)

if __name__ == "__main__":
   
   
    plt.close('all')
   
    #example: fit a GP (with noisy observations)
    X = np.array([1., 3., 5., 6., 7., 8.]).reshape(-1,1)
    y = f(X).ravel()
    dy = 0.5 + 1.0*np.random.random(y.shape)  #in [0.5, 1.5] <- std deviation per point
    y = y + np.random.normal(0, dy)  #0-mean noise with variable std in [0.5, 1.5]
    gp = GaussianProcess(corr='cubic', nugget = (dy / y)**2, theta0=1e-1, thetaL=1e-3, thetaU=1, random_start=100, verbose=True)
    gp.fit(X, y)  #ML est
    gp.get_params()
        
    Xt = np.array(np.linspace(np.min(X)-10,np.max(X)+10,1000)).reshape(-1,1)
    y_pred, MSE = gp.predict(Xt, eval_MSE=True)
    sigma = np.sqrt(MSE)
    
    plt.figure()
    plt.plot(Xt, f(Xt), color='k', lw=2.0, label = 'x sin(x) ground truth')
    plt.plot(X, y, 'r+', markersize=20, lw=2.0, label = 'observations')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')    
    plt.plot(Xt, y_pred, color = 'g', linestyle = '--', lw=1.5, label = 'GP prediction')
    plt.fill(np.concatenate([Xt, Xt[::-1]]), np.concatenate([y_pred-1.96*sigma, (y_pred+1.96*sigma)[::-1]]), alpha = 0.5, label = '95% conf interval')
    plt.title('GP regression')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.legend()
    plt.show()
            
    #fit a GP to market data
    #load data     
    start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)    
    spy = data.DataReader("SPY", 'google', start, end)
    
    spy_price = np.array(spy['Close'].values).reshape(-1,1)
    spy_volume = np.array(spy['Volume'].values).reshape(-1,1)
    spy_obs = np.hstack([spy_price, spy_volume])
                
    #X = np.random.rand(np.size(spy_price)).reshape(-1,1)        
    X = np.array(range(np.size(spy_price))).reshape(-1,1)
    y = spy_price.ravel()
    dy = 10*spy.std()['Close']
    spy_gp = GaussianProcess(corr='cubic', nugget = (dy/y)**2, theta0=1e-1, thetaL=1e-3, thetaU=1e3, random_start=100, verbose=True)
    spy_gp.fit(X,y)
    
    spy_gp.get_params()
        
    Xt = np.array(np.linspace(np.min(X)-10,np.max(X)+10,1000)).reshape(-1,1)
    y_pred, MSE = spy_gp.predict(Xt, eval_MSE=True)
    sigma = np.sqrt(MSE)            

    f = plt.figure()
    plt.plot(X, y, 'r-', markersize=20, lw=2.0, label = 'SPY price, USD')
    #plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')    
    plt.plot(Xt, y_pred, color = 'g', linestyle = '--', lw=1.5, label = 'GP prediction')
    plt.fill(np.concatenate([Xt, Xt[::-1]]), np.concatenate([y_pred-1.96*sigma, (y_pred+1.96*sigma)[::-1]]), alpha = 0.5, label = '95% conf interval')
    plt.title('GP regression')
    plt.xlabel('time, days')
    plt.ylabel('S&P500 price, USD')
    plt.grid(True)
    plt.legend()
    plt.show()