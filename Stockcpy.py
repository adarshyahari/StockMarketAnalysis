import seaborn as sns
import matplotlib.pyplot as plt
from altair.vegalite.v4.schema.core import Mark
import altair as alt
import streamlit as st
import pandas_datareader.data as web
import pandas_datareader as pdr
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# Python Data Analysis imports
import datetime as dt

# Streamlit imports
alt.renderers.enable('mimetype')
alt.renderers.enable('html')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Visualisation imports

# Setting the end date to today
end = dt.datetime.now()
# Get dates for 1 month, 6 month, 1 year
year = dt.datetime(end.year-1, end.month, end.day)
sixmo = dt.datetime(end.year, end.month-6, end.day)
onemo = dt.datetime(end.year, end.month-1, end.day)

# Global stock variable
itstocks = ['TCS.NS', 'WIPRO.NS', 'INFY.NS', 'PERSISTENT.NS',
            'COFORGE.NS', 'LT.NS', 'TECHM.NS', 'MPHASIS.NS', 'HCLTECH.NS', 'OFSS.NS']
phstocks = ['AJANTPHARM.NS', 'ALKEM.NS', 'CIPLA.NS', 'ERIS.NS', 'IPCALAB.NS',
            'JBCHEPHARM.NS', 'SUNPHARMA.NS', 'GLAND.NS', 'DIVISLAB.NS', 'SYNGENE.NS']
austocks = ['BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'M&M.NS', 'BALKRISIND.NS',
            'ENDURANCE.NS', 'EXIDEIND.NS', 'MAHINDCIE.NS', 'MINDAIND.NS', 'MOTHERSUMI.NS']
enstocks = ['BPCL.NS', 'RELIANCE.NS', 'COALINDIA.NS', 'GAIL.NS',
            'GSPL.NS', 'IEX.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'NHPC.NS']


@st.cache(persist=True)
def get_data(stocks, start, end):
    data = pdr.get_data_yahoo(stocks, start, end)
    return data


@st.cache(persist=True)
def value_risk():
    runs = 1000
    simulations = np.zeros(runs)
    for run in range(runs):
        simulations[run] = stock_monte_carlo(
            start_price, days, mu, sigma)[days-1]
    return simulations, np.percentile(simulations, 1)


@st.cache(persist=True)
def sharpe_ratio():
    for portfolio in range(number_of_portfolios):
        # generate random weight of length of number of stocks
        weights = np.random.random_sample(10)
        weights = weights / np.sum(weights)
        portfolio_weights.append(weights)

        # Calculating the annualised portfolio returns
        annualize_return = np.sum((return_stocks.mean() * weights) * 252)
        portfolio_returns.append(annualize_return)

        # Calculating risk using variance
        matrix_covariance_portfolio = (return_stocks.cov())*252
        portfolio_variance = np.dot(weights.T, np.dot(
            matrix_covariance_portfolio, weights))
        portfolio_standard_deviation = np.sqrt(portfolio_variance)
        portfolio_risk.append(portfolio_standard_deviation)

        # Calculating sharpe_ratio using return and risk
        sharpe_ratio = ((annualize_return - RF)/portfolio_standard_deviation)
        sharpe_ratio_port.append(sharpe_ratio)
    return portfolio_risk, portfolio_returns, sharpe_ratio_port, portfolio_weights


tracker_select = st.sidebar.selectbox(
    label='Select tracker',
    options=['IT', 'Pharma', 'Auto', 'Energy']
)

if tracker_select == 'IT':
    title = 'IT Tracker'
    caption = 'Companies to efficiently track and invest in the energy sector'
    stocks = itstocks
    trackeroptions = ['TCS', 'WIPRO', 'INFY', 'PERSISTENT',
                      'COFORGE', 'LT', 'TECHM', 'MPHASIS', 'HCLTECH', 'OFSS']
    stocknames = ['Tata Consultancy', 'Wipro Ltd', 'Infosys Ltd', 'Persistent Systems Ltd', 'Coforge Ltd', 'Larsen & Toubro Infotech Ltd',
                  'Tech Mahindra Ltd', 'Mphasis Ltd', 'HCL Technologies Ltd', 'Oracle Financial Services Software Ltd']
    stocktype = ['IT Services & Consulting', 'IT Services & Consulting', 'IT Services & Consulting', 'IT Services & Consulting', 'IT Services & Consulting',
                 'IT Services & Consulting', 'IT Services & Consulting', 'IT Services & Consulting', 'IT Services & Consulting', 'Software Services']
    typecount = [9, 1]

elif tracker_select == 'Pharma':
    title = 'Pharma Tracker'
    caption = 'Companies to efficiently track and invest in the pharma sector'
    stocks = phstocks
    trackeroptions = ['AJANTPHARM', 'ALKEM', 'CIPLA', 'ERIS', 'IPCALAB',
                      'JBCHEPHARM', 'SUNPHARMA', 'GLAND', 'DIVISLAB', 'SYNGENE']
    stocknames = ['Ajanta Pharma Ltd', 'Alkem Laboratories Ltd', 'Cipla Ltd', 'Eris Lifesciences Ltd', 'IPCA Laboratories Ltd',
                  'J B Chemicals and Pharmaceuticals Ltd', 'Sun Pharmaceutical Industries Ltd', 'Gland Pharma Ltd', "Divi's Laboratories Ltd", 'Syngene International Ltd']
    stocktype = ['Pharmaceuticals', 'Pharmaceuticals', 'Pharmaceuticals', 'Pharmaceuticals', 'Pharmaceuticals',
                 'Pharmaceuticals', 'Pharmaceuticals', 'Life Sciences Tools & Services', 'Life Sciences Tools & Services']
    typecount = [8, 2]

elif tracker_select == 'Auto':
    title = 'Auto Tracker'
    caption = 'Companies to efficiently track and invest in the auto sector'
    stocks = austocks
    trackeroptions = ['BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'M&M',
                      'BALKRISIND', 'ENDURANCE', 'EXIDEIND', 'MAHINDCIE', 'MINDAIND', 'MOTHERSUMI']
    stocknames = ['Bajaj Auto Ltd', 'Eicher Motors Ltd', 'Hero MotoCorp Ltd', 'Mahindra and Mahindra Ltd', 'Balkrishna Industries Ltd',
                  'Endurance Technologies Ltd (CN)', 'Exide Industries Ltd', 'Mahindra CIE Automotive Ltd', 'Minda Industries Ltd', 'Motherson Sumi Systems Ltd']
    stocktype = ['Automobiles', 'Automobiles', 'Automobiles', 'Automobiles',
                 'Tires & Rubber', 'Auto Parts', 'Auto Parts', 'Auto Parts', 'Auto Parts', 'Auto Parts']
    typecount = [4, 1, 5]

elif tracker_select == 'Energy':
    title = 'Energy Tracker'
    caption = 'Companies to efficiently track and invest in the energy sector'
    stocks = enstocks
    trackeroptions = ['BPCL', 'RELIANCE', 'COALINDIA', 'GAIL',
                      'GSPL', 'IEX', 'NTPC', 'ONGC', 'POWERGRID', 'NHPC']
    stocknames = ['Bharat Petroleum Corporation Ltd', 'Reliance Industries Ltd', 'Coal India Ltd',
                  'GAIL (India) Ltd', 'Gujarat State Petronet Ltd', 'Indian Energy Exchange Ltd', 'NTPC Ltd', 'Oil and Natural Gas Corporation Ltd', 'Power Grid Corporation of India Ltd', 'NHPC Ltd']
    stocktype = ['Oil & Gas - Refining & Marketing', 'Oil & Gas - Refining & Marketing', 'Coal', 'Gas Distribution', 'Gas Distribution',
                 'Power Trading', 'Power Generation', 'Oil & Gas - Exploration & Production', 'Power Transmission & Distribution', 'Power Generation']
    typecount = [2, 1, 2, 1, 2, 1, 1]

module_select = st.sidebar.selectbox(
    label='Select module',
    options=['Tracker Overview', 'Optimized Portfolios',
             'Equal Weightage', 'Stock specific']
)
if module_select == 'Tracker Overview':
    try:
        st.title(title)
        st.caption(caption)
        stock_table = np.array(stocknames)
        ticker_table = np.array(stocks)
        type_table = np.array(stocktype)
        table = [stock_table, type_table, ticker_table]
        table_dfs = pd.DataFrame(table)
        table_dfs = table_dfs.T
        # Rename the columns:
        table_dfs.columns = ['Company', 'Type', 'Ticker']
        st.table(table_dfs)

        names = set(type_table)
        size = np.array(typecount)
        plt.title('Segment Composition', weight="bold")
        my_circle = plt.Circle((0, 0), 0.7, color='white')
        colors = ("#CBF2B8", "#FFA1CA", "#AA98F0",
                  '#FDF8AF', '#F5C1C1', '#A6EDFF', '#ffdca9')
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.pie(size, labels=names, colors=colors)
        st.pyplot()

        df = get_data(stocks, year, end)

        # Closing Prices
        adj = df['Adj Close']
        rets_df = adj.pct_change()
        mean_df = rets_df.mean()

        # Market Cap
        market_data = []
        for ticker in stocks:
            try:
                market_data.append(web.get_quote_yahoo(ticker)['marketCap'])
            except:
                print('Error with: ', ticker)
        mc = pd.concat(market_data, axis=0)

        lcount = 0
        mcount = 0
        scount = 0
        for m in mc:
            if m > 2000000000000:
                lcount = lcount+1
            elif m < 500000000000:
                scount = scount+1
            else:
                mcount = mcount+1

        st.subheader('Market Cap of Stocks')
        col1, col2 = st.columns(2)
        with col1:
            names = ['Large Cap', 'Mid Cap', 'Small Cap']
            size = np.array(
                [(lcount/10)*100, (mcount/10)*100, (scount/10)*100])

            # Create a circle at the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='white')
            colors = ("#957DAD", "#FEC8D8", "#FFDFD3")
            plt.figure(figsize=(2, 2))
            p = plt.gcf()
            p.gca().add_artist(my_circle)

            plt.pie(size, labels=names, colors=colors)
            st.pyplot()
        with col2:
            st.bar_chart(mc, width=600, height=400, use_container_width=False)

        # Correlation Heat Map of stocks in portfolio
        st.subheader('Correlation between stocks')
        corr = rets_df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            fig, ax = plt.subplots()
            ax = sns.heatmap(corr, cmap="YlGnBu", annot=True,
                             linewidths=1, linecolor='white', square=True, mask=mask)
            st.write(fig)

        # Risk vs. Return of stocks in portfolio
        st.subheader('Risk vs. Return of stock')
        rets = rets_df.dropna()
        plt.figure(figsize=(12, 10))
        plt.scatter(rets.mean(), rets.std(), s=25)
        plt.xlabel('Expected Return')
        plt.ylabel('Risk')
        # For adding annotatios in the scatterplot
        for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
            plt.annotate(
                label,
                fontsize=10,
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points', ha='right', va='bottom'
            )
        st.pyplot()

    except Exception as e:
        print(e)

elif module_select == 'Stock specific':
    try:
        option = st.sidebar.selectbox(
            label='Select Stock',
            options=trackeroptions)
        for i in range(0, 11):
            if option == stocks[i][:-3]:
                try:
                    # Get stock
                    st.title(stocknames[i])
                    st.caption(stocktype[i])
                    stk = get_data(stocks[i], year, end)
                    length = int(len(stk))
                    stkone = get_data(stocks[i], onemo, end)

                    pe_data = web.get_quote_yahoo(stocks[i])['trailingPE']
                    mc_data = web.get_quote_yahoo(stocks[i])['marketCap']

                    tickers = ['Adj Close', 'Open', 'High', 'Low']
                    col1, col2, col3, col4 = st.columns(4)
                    tickercol = [col1, col2, col3, col4]
                    for t in range(4):
                        l = stk[tickers[t]][-1]
                        f = stk[tickers[t]][-2]
                        inc = l-f
                        iper = inc/f * 100
                        tickercol[t].metric(tickers[t], "₹{}".format(
                            round(l, 2)), "{}%".format(round(iper, 2)))
                    col1.metric("PE Ratio", "{}".format(
                        round(float(pe_data), 2)))
                    col2.metric("Market Cap", "{}LCr".format(
                        round(float(mc_data/1000000000000), 2)))
                    col3.metric('52-week High',
                                '₹{}'.format(round(stk['High'].max(), 2)))
                    col4.metric('52-week Low',
                                '₹{}'.format(round(stk['Low'].min(), 2)))

                    # Adjusted Closing price
                    st.subheader('Adjusted Closing Price')
                    col1, col2, col3, col4, d1, d2, d3, d4, d5, d6, d7, d8 = st.columns(
                        12)
                    with col1:
                        b1 = st.button('1Y')
                    with col2:
                        b2 = st.button('6M')
                    with col3:
                        b3 = st.button('1M')
                    with col4:
                        b4 = st.button('5D')

                    adjclose = pd.DataFrame(stk['Adj Close'])
                    adjsix = pd.DataFrame(stk['Adj Close'].tail(int(length/2)))

                    if b1:
                        st.area_chart(adjclose, width=800, height=400,
                                      use_container_width=False)
                    elif b2:
                        st.area_chart(adjsix, width=800, height=400,
                                      use_container_width=False)
                    elif b3:
                        adjone = pd.DataFrame(stkone['Adj Close'])
                        st.area_chart(adjone, width=800, height=400,
                                      use_container_width=False)
                    elif b4:
                        adjfive = pd.DataFrame(stkone['Adj Close'].tail(5))
                        st.area_chart(adjfive, width=800, height=400,
                                      use_container_width=False)

                    # Volume Traded
                    st.subheader('Volume Traded over time')
                    volume = pd.DataFrame(stk['Volume'])
                    st.area_chart(volume, width=800, height=400,
                                  use_container_width=False)

                    # Moving average of stock for 10,20,50 days
                    st.subheader('Moving average of stock')
                    ma_day = [10, 20, 50]
                    for ma in ma_day:
                        column_name = "MA for %s days" % (str(ma))
                        stk[column_name] = adjclose.rolling(
                            window=ma, center=False).mean()
                    plt.figure(figsize=(2, 3))
                    st.line_chart(stk[['Adj Close', 'MA for 10 days',
                                       'MA for 20 days', 'MA for 50 days']], width=800, height=400,
                                  use_container_width=False)

                    # Daily return of stock in %
                    st.subheader('Daily Return')
                    dlrt = adjclose.pct_change()
                    st.metric('Daily Return Average', "{}%".format(
                        round(float(dlrt.mean())*100, 2)))
                    st.line_chart(dlrt, width=800, height=400,
                                  use_container_width=False)

                    # Daily return in histogram with KDE
                    plt.figure(figsize=(8, 3))
                    sns.distplot(dlrt.dropna(), bins=300, color='red')
                    st.pyplot()

                    # Monte Carlo Function
                    days = 365
                    dt = 1/365
                    mu = dlrt.mean()
                    sigma = dlrt.std()

                    def stock_monte_carlo(start_price, days, mu, sigma):
                        price = np.zeros(days)
                        price[0] = start_price
                        shock = np.zeros(days)
                        drift = np.zeros(days)
                        for x in range(1, days):
                            shock[x] = np.random.normal(
                                loc=mu*dt, scale=sigma*np.sqrt(dt))
                            drift[x] = mu * dt
                            price[x] = price[x-1] + \
                                (price[x-1] * (drift[x]+shock[x]))
                        return price

                    start_price = stk['Open'].head()[0]

                    # Value at Risk
                    st.subheader('Value at Risk')

                    monte = st.checkbox(
                        "Show Monte Carlo Analysis for 100 runs")
                    if monte:
                        # Visualize using scatter plot in matplotlib
                        plt.figure(figsize=(12, 7))
                        plt.xlabel('Days')
                        plt.ylabel('Price')
                        plt.title('Monte Carlo Analysis for {}'.format(stocknames[i]),
                                  weight="bold")

                        for run in range(100):
                            plt.plot(stock_monte_carlo(
                                start_price, days, mu, sigma))
                        st.pyplot()

                    simulations, q = value_risk()
                    st.caption('If you invested ₹{} exactly one year ago, the maximum loss you would have incurred in a year is ₹{}'.format(
                        round(start_price, 2), round(start_price-q, 2)))
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric('Start Price', '₹{}'.format(
                        round(start_price, 2)))
                    col2.metric('Mean Final Price', '₹{}'.format(
                        round(simulations.mean(), 2)))
                    col4.metric('Lossy Final Price', '₹{}'.format(round(q, 2)))
                    col3.metric('Value at risk', '₹{}'.format(
                        round(start_price-q, 2)))

                    plt.figure(figsize=(16, 7))
                    plt.hist(simulations, bins=200)
                    plt.figtext(0.15, 0.6, "q(0.99): Rs%.2f" % q)
                    plt.axvline(x=q, linewidth=4, color='r')
                    st.pyplot()

                    # Sentiment Analysis
                    st.subheader(
                        'Sentiment Analysis of Stock for 7-day period')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image('download (1).png',
                                 caption='Closing Price of stock', width=None)
                    with col2:
                        st.image('download (2).png',
                                 caption='Sentiment of news on equity', width=None)

                except Exception as e:
                    st.write(e)
    except Exception as e:
        print(e)

elif module_select == 'Equal Weightage':
    try:
        st.title(title)
        st.caption(caption)
        col1, col2, col3 = st.columns(3)

        initial_weight = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        df = get_data(stocks, year, end)

        # Average daily return of stocks
        return_stocks = df['Adj Close'].pct_change()
        Mean = (return_stocks.mean())*100
        st.table(Mean)
        st.write(Mean.idxmax(), 'has the highest daily return of',
                 round((Mean.max()), 2), '%')

        # Average daily return of portfolio
        allocated_daily_returns = (initial_weight * Mean)
        portfolio_return = np.sum(allocated_daily_returns)
        changeper = "{}%".format(round(portfolio_return*100, 2))
        col1.metric("Daily Change", changeper)

        # Cumulative return of stocks over a year
        return_stocks['Portfolio Return'] = return_stocks.dot(initial_weight)
        cumulative_returns_daily = ((1+return_stocks).cumprod())*100
        st.table(cumulative_returns_daily.tail(1).max())
        st.write('The biggest driver of return is', cumulative_returns_daily.tail(1).max().idxmax(
        ), 'with a cumulative return of', round(cumulative_returns_daily.tail(1).max().max(), 2), '%')

        # Cumulative return of portfolio over a year
        plt.figure(figsize=(12, 7))
        plt.title("Portfolio Daily Returns", weight='bold')
        cumulative_returns_daily['Portfolio Return'].plot()
        st.pyplot()
        lastret = cumulative_returns_daily['Portfolio Return'][-1]
        firstret = cumulative_returns_daily['Portfolio Return'][1]
        incret = lastret-firstret
        incper = incret/firstret * 100
        incper = "{}%".format(round(incper, 2))
        col2.metric("1Y Cumulative Return", incper)

        # Portfolio risk
        matrix_covariance_portfolio = return_stocks.iloc[:, :-1]
        matrix_covariance_portfolio = (matrix_covariance_portfolio.cov())*252
        portfolio_variance = np.dot(initial_weight.T, np.dot(
            matrix_covariance_portfolio, initial_weight))
        portfolio_risk = np.sqrt(portfolio_variance)
        riskper = "{}%".format(round(portfolio_risk*100, 2))
        col3.metric("Risk Percentage", riskper)

    except Exception as e:
        print(e)
elif module_select == 'Optimized Portfolios':
    try:
        st.title(title)
        st.caption(caption)
        df = get_data(stocks, year, end)
        return_stocks = df['Close'].pct_change()
        number_of_portfolios = 2000

        RF = 0
        portfolio_returns = []
        portfolio_risk = []
        sharpe_ratio_port = []
        portfolio_weights = []

        # Converting list into numpy arrays
        portfolio_risk, portfolio_returns, sharpe_ratio_port, portfolio_weights = sharpe_ratio()

        portfolio_risk = np.array(portfolio_risk)
        portfolio_returns = np.array(portfolio_returns)
        sharpe_ratio_port = np.array(sharpe_ratio_port)

        agree = st.checkbox("Show generated portfolio scatterplot")
        if agree:
            # Visualize using scatter plot in matplotlib
            plt.figure(figsize=(12, 7))
            plt.scatter(portfolio_risk, portfolio_returns,
                        c=portfolio_returns / portfolio_risk)
            plt.xlabel('volatility')
            plt.ylabel('returns')
            plt.colorbar(label='Sharpe ratio')
            plt.title('Sharpe Ratio for random portfolios', weight="bold")
            st.pyplot()

            source = pd.DataFrame({
                'Risk': portfolio_risk,
                'Return': portfolio_returns
            })
            alt.Chart(source).mark_circle(size=60).encode(
                x='Risk',
                y='Return',
                tooltip=['Risk', 'Return']
            ).interactive()

        # We consider the metrics: return, risk, sharpe ratio, and weights
        porfolio_metrics = [portfolio_returns, portfolio_risk,
                            sharpe_ratio_port, portfolio_weights]

        # from Python list we create a Pandas DataFrame
        portfolio_dfs = pd.DataFrame(porfolio_metrics)
        portfolio_dfs = portfolio_dfs.T

        # Rename the columns:
        portfolio_dfs.columns = [
            'Port Returns', 'Port Risk', 'Sharpe Ratio', 'Portfolio Weights']

        # convert from object to float the first three columns.
        for col in ['Port Returns', 'Port Risk', 'Sharpe Ratio']:
            portfolio_dfs[col] = portfolio_dfs[col].astype(float)

        # Portfolio with the highest Sharpe Ratio can be found using idxmax
        Highest_sharpe_port = portfolio_dfs.iloc[portfolio_dfs['Sharpe Ratio'].idxmax(
        )]

        # Portfolio with the minimum risk can be found using idxmin
        min_risk = portfolio_dfs.iloc[portfolio_dfs['Port Risk'].idxmin()]

        st.header('High Return Portfolio')

        col1, col2, col3 = st.columns(3)
        hrreturn = "{}%".format(
            round(Highest_sharpe_port['Port Returns']*100, 2))
        hrrisk = "{}%".format(round(Highest_sharpe_port['Port Risk']*100, 2))
        hrsharpe = "{}".format(round(Highest_sharpe_port['Sharpe Ratio'], 2))
        col1.metric("Return Percentage", hrreturn)
        col2.metric("Risk Percentage", hrrisk)
        col3.metric("Sharpe Ratio", hrsharpe)

        # st.write('Portfolio with highest sharpe ratio of',round(Highest_sharpe_port['Sharpe Ratio'],2), 'has a return of',
        #     round(Highest_sharpe_port['Port Returns']*100, 2), '% and risk of', round(Highest_sharpe_port['Port Risk']*100, 2),"%")

        j = 0
        data = []
        for i in stocks:
            temp = Highest_sharpe_port['Portfolio Weights'][j]*100
            data.append([i, round(temp, 2)])
            j = j+1
        sharpe = pd.DataFrame(
            data,
            columns=['Stock', 'Weightage in %']
        )
        st.table(sharpe)

        st.header('Low Risk Portfolio')

        col1, col2, col3 = st.columns(3)
        mrreturn = "{}%".format(round(min_risk['Port Returns']*100, 2))
        mrrisk = "{}%".format(round(min_risk['Port Risk']*100, 2))
        mrsharpe = "{}".format(round(min_risk['Sharpe Ratio'], 2))
        col1.metric("Return Percentage", mrreturn)
        col2.metric("Risk Percentage", mrrisk)
        col3.metric("Sharpe Ratio", mrsharpe)

    #     st.write('Portfolio with Minimum Risk of',round(min_risk['Port Risk']*100, 2),"% has a sharpe ratio of ",round(min_risk['Sharpe Ratio'],2), 'and a return of',
    #   round(min_risk['Port Returns']*100, 2), '%')

        j = 0
        data = []
        for i in stocks:
            temp = min_risk['Portfolio Weights'][j]*100
            data.append([i, round(temp, 2)])
            j = j+1
        minrisk = pd.DataFrame(
            data,
            columns=['Stock', 'Weightage in %']
        )
        st.table(minrisk)

    except Exception as e:
        st.write(e)
