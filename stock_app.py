import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import yfinance as yf
from PIL import Image
import time
from pandas_datareader import data
from datetime import date

timestr = time.strftime("%Y%m%d-%H%M%S")


@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

# Fxn to Download Result
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "SP500_{}_.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

def make_downloadable_allocation(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "allocation_{}_.csv".format(timestr)
    st.markdown("### ** ⬇️ Download CSV file **")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

def get_sharpe(data):
    
    test = data['Adj Close']
    log_ret = np.log(test/test.shift(1))
    
    np.random.seed(101)

    num_ports = 10000
    all_weights = np.zeros((num_ports,len(test.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):

        ## Weights
        weights = np.array(np.random.random(test.shape[1]))
        weights = weights/np.sum(weights)

        ## Save Weights
        all_weights[ind,:] = weights

        ## Expected Return
        ret_arr[ind] = np.sum((log_ret.mean() * weights)* 252)

        ## Expected Volatility
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252, weights)))

        ## Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    
    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
        
    return sharpe_arr.max(), all_weights[sharpe_arr.argmax(),:], max_sr_ret, max_sr_vol, ret_arr, vol_arr, sharpe_arr

def plot_frontier(volatility_array, returns_array, sharpe_array, max_volatility, max_return):
    fig = plt.figure(figsize = (10,5))

    plt.scatter(volatility_array, returns_array, c=sharpe_array, cmap='plasma')
    plt.scatter(max_volatility, max_return, c='green', s=50, edgecolors='black', label = 'Optimal')

    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(alpha = 0.2)
    plt.title('Efficient Frontier')

    st.pyplot(fig)

def plot_close(data):
    fig = plt.figure(figsize = (10,5))

    test = data['Adj Close']

    plt.plot(test, label = list(test.columns))
    plt.legend()

    plt.grid(alpha = 0.2)
    plt.title('Selected Ticker Historic Prices')

    plt.xlabel('Year')
    plt.ylabel('USD')

    st.pyplot(fig)

def main():

    image = Image.open('sp500.jpg')

    st.image(image, width = 500)

    st.title('S&P 500 App')

    st.markdown("""
    This app retrieves the list of the **S&P 500** (from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)) and its corresponding **stock closing price** (year-to-date)!
    """)

    df = load_data()
    
    sector = df.groupby('GICS Sector')

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df['GICS Sector'].unique())
    selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

    # Filtering data
    df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]

    st.header('Display Companies in Selected Sector')
    st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
    st.dataframe(df_selected_sector)

    with st.expander('Download CSV'):
        make_downloadable(df_selected_sector)

    st.header('Optimal Portfolio Allocation (For Educational Purposes Only)')

    st.markdown(""" The algorithm uses a monte carlo simulation approach and tests 10,000 different allocations. 
    Choose a set of sectors and ticker symbols of interest and wait to view the optimal allocation results below.""")

    tickers = list(df_selected_sector['Symbol'])
    my_tickers = st.sidebar.multiselect('Tickers', tickers)

    today = date.today()
    d1 = today.strftime("%Y/%m/%d")

    if my_tickers != []:
        df = data.DataReader(my_tickers, 'yahoo', start='2000/01/01', end=d1)

        sr, w, ret, vol, ret_ar, vol_ar, s_ar = get_sharpe(df)
        
        c1, c2 = st.columns(2)
        with c1:
            with st.expander('Best Expected Return (%)'):
                ret_str = str(round(ret * 100, 2))
                st.write(ret_str)

            with st.expander('Sharpe Ratio'):
                st.write(sr)
        
        with c2:
            with st.expander('Best Expected Volatility (%)'):
                vol_str = str(round(vol * 100, 2))
                st.write(vol_str)

            with st.expander('Portfolio Allocation'):
                tick = my_tickers
                al = list(w * 100)
                result = pd.DataFrame(list(zip(tick, al)), columns =['Ticker Choice', 'Allocation (%)'])
                st.dataframe(result)

        with st.expander('Show Efficient Frontier'):
            plot_frontier(vol_ar, ret_ar, s_ar, vol, ret)
        
        with st.expander('Show Tickers Historical Prices'):
            plot_close(df)

        with st.expander('Download Portfolio Allocation CSV'):
            make_downloadable_allocation(result)
        
if __name__ == '__main__':
    main()
