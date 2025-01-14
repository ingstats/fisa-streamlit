import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec

# 한글 폰트 설정 (시스템에 맞게 변경 가능)
plt.rcParams['font.family'] = 'Malgun Gothic'

@st.cache_data
def get_stock_info():
    base_url = "http://kind.krx.co.kr/corpgeneral/corpList.do"    
    method = "download"
    url = f"{base_url}?method={method}"   
    df = pd.read_html(url, header=0, encoding='euc-kr')[0]
    df['종목코드'] = df['종목코드'].apply(lambda x: f"{x:06d}")     
    df = df[['회사명','종목코드']]
    return df

def get_ticker_symbol(company_name):     
    df = get_stock_info()
    matched = df[df['회사명'] == company_name]
    if matched.empty:
        return None 
    return matched['종목코드'].values[0]

# 페이지 설정
st.set_page_config(page_title="주식 차트", layout="wide")
st.title("📈 주식 차트")

st.markdown(
    """
    주식 종목명을 입력하고 조회 기간을 설정하여 주가 차트를 확인하세요.
    사이드바에서 구매 정보 및 추가 그래프 옵션을 선택할 수 있습니다.
    """,
    unsafe_allow_html=True
)

# 사이드바 입력 항목들
with st.sidebar:
    st.header("회사 정보 입력")
    stock_name = st.text_input('회사 이름', value="삼성전자").upper()
    st.caption("예: 삼성전자, 현대자동차 등")
    
    st.header("조회 기간 설정")
    date_range = st.date_input(
        "시작일 - 종료일",
        [datetime.date(2019, 1, 1), datetime.date(2025, 1, 13)]
    )

    with st.expander("구매 정보 및 손익 계산 (선택사항)", expanded=False):
        calculate_profit = st.checkbox("구매 정보 입력 및 손익 계산하기")
        if calculate_profit:
            st.subheader("구매 정보 입력")
            purchase_date = st.date_input("구매일", value=datetime.date(2020, 1, 1))
            purchase_qty = st.number_input("구매 수량", min_value=1, value=10)

    with st.expander("추가 그래프 옵션", expanded=False):
        show_candlestick = st.checkbox("캔들스틱 차트 보기")
        show_moving_average = st.checkbox("이동 평균선 보기")
        show_rsi = st.checkbox("RSI 차트 보기")
        show_macd = st.checkbox("MACD 차트 보기")
        show_bollinger = st.checkbox("Bollinger Bands 보기")
        show_volume = st.checkbox("거래량 차트 보기")

    clicked = st.button("조회")

if clicked:
    ticker_symbol = get_ticker_symbol(stock_name)
    if ticker_symbol is None:
        st.error("존재하지 않는 회사명입니다. 정확한 회사명을 입력해주세요.")
    else:
        with st.spinner("데이터를 불러오는 중..."):
            start_p = date_range[0]
            end_p = date_range[1] + datetime.timedelta(days=1)
            df = fdr.DataReader(ticker_symbol, start_p, end_p, exchange="KRX")
            df.index = pd.to_datetime(df.index.date)

        st.subheader(f"[{stock_name}] 관련 정보")

        # 기술적 분석 지표 계산
        delta = df['Close'].diff()
        gain = delta.copy()
        gain[gain < 0] = 0
        loss = -delta.copy()
        loss[loss < 0] = 0
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = df['MA20'] + 2 * df['BB_std']
        df['Lower_BB'] = df['MA20'] - 2 * df['BB_std']

        if 'calculate_profit' in locals() and calculate_profit:
            if purchase_date < start_p or purchase_date > date_range[1]:
                st.warning("구매일이 조회 기간 범위 밖입니다. 손익 계산을 할 수 없습니다.")
            else:
                purchase_day = min(df.index, key=lambda d: abs(d - pd.to_datetime(purchase_date)))
                purchase_price = df.loc[purchase_day]['Close']
                current_price = df['Close'].iloc[-1]
                profit_per_share = current_price - purchase_price
                total_profit = profit_per_share * purchase_qty

                st.markdown("### 📈 손익 계산 결과")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="구매일", value=str(purchase_day.date()))
                    st.metric(label="구매가", value=f"{purchase_price:,.2f} 원")
                    st.metric(label="현재가", value=f"{current_price:,.2f} 원")
                with col2:
                    st.metric(label="주당 손익", value=f"{profit_per_share:,.2f} 원")
                    st.metric(label=f"총 손익 ({purchase_qty}주)", value=f"{total_profit:,.2f} 원")

        # 주가 및 거래량 차트 (가격과 거래량을 함께 시각화)
        fig_vol, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(15, 8), 
                                                 sharex=True, 
                                                 gridspec_kw={'height_ratios': [3, 1]})
        ax_price.plot(df.index, df['Close'], label='종가')
        if show_moving_average:
            df['MA50'] = df['Close'].rolling(window=50).mean()
            ax_price.plot(df.index, df['MA20'], label='20일 이동 평균')
            ax_price.plot(df.index, df['MA50'], label='50일 이동 평균')
        if show_bollinger:
            ax_price.fill_between(df.index, df['Lower_BB'], df['Upper_BB'], color='grey', alpha=0.3, label='Bollinger Bands')
        ax_price.set_title("주가(종가) 및 Bollinger Bands")
        ax_price.set_ylabel("가격(원)")
        ax_price.legend(fontsize=12)

        if show_volume:
            ax_vol.bar(df.index, df['Volume'], width=1.0, color='blue')
            ax_vol.set_ylabel("거래량")
        else:
            ax_vol.set_visible(False)

        plt.xticks(rotation=45)

        st.markdown(
            """
            ### 📊 주가 및 거래량 차트
            - **종가 (Close)**: 주식의 최종 거래 가격을 나타냅니다.
            - **이동 평균선 (MA20, MA50)**: 일정 기간 동안의 평균 가격을 계산하여 추세를 파악하는 데 도움을 줍니다.
                - **20일 이동 평균선 (MA20)**: 단기적인 가격 추세를 나타냅니다.
                - **50일 이동 평균선 (MA50)**: 중기적인 가격 추세를 나타냅니다.
            - **Bollinger Bands**: 주가의 변동성을 측정하기 위한 지표로, 주가가 상단 밴드와 하단 밴드 사이에서 움직이는 경향을 보입니다.
            - **거래량 (Volume)**: 특정 기간 동안의 주식 거래량을 나타내며, 시장의 활발함을 보여줍니다.
            """
        )

        st.pyplot(fig_vol)

        # 별도의 주가 및 Bollinger Bands 그래프
        fig_bollinger, ax_bollinger = plt.subplots(figsize=(15, 5))
        ax_bollinger.plot(df.index, df['Close'], label='종가')
        if show_bollinger:
            ax_bollinger.fill_between(df.index, df['Lower_BB'], df['Upper_BB'], color='grey', alpha=0.3, label='Bollinger Bands')
        ax_bollinger.set_title("주가(종가) 및 Bollinger Bands")
        ax_bollinger.set_ylabel("가격(원)")
        ax_bollinger.legend(fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig_bollinger)

        # RSI 차트
        if show_rsi:
            fig_rsi, ax_rsi = plt.subplots(figsize=(15, 3))
            ax_rsi.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(70, color='red', linestyle='--')
            ax_rsi.axhline(30, color='green', linestyle='--')
            ax_rsi.set_title("RSI (Relative Strength Index)")
            ax_rsi.set_ylabel("RSI")
            ax_rsi.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_rsi)

            st.markdown(
                """
                ### 🔍 RSI (Relative Strength Index)
                - **RSI**는 주식의 과매수 또는 과매도 상태를 나타내는 지표입니다.
                - **70 이상**: 과매수 상태로, 가격이 과도하게 상승하여 조정이 일어날 가능성이 있습니다.
                - **30 이하**: 과매도 상태로, 가격이 과도하게 하락하여 반등할 가능성이 있습니다.
                """
            )

        # MACD 차트
        if show_macd:
            fig_macd, ax_macd = plt.subplots(figsize=(15, 3))
            ax_macd.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax_macd.plot(df.index, df['Signal'], label='Signal', color='orange')
            ax_macd.legend()
            ax_macd.set_title("MACD (Moving Average Convergence Divergence)")
            plt.xticks(rotation=45)
            st.pyplot(fig_macd)

            st.markdown(
                """
                ### 📈 MACD (Moving Average Convergence Divergence)
                - **MACD**는 주가의 추세와 모멘텀을 파악하기 위한 지표입니다.
                - **MACD 선**과 **Signal 선**의 교차점을 통해 매수/매도 신호를 확인할 수 있습니다.
                    - **MACD 선이 Signal 선을 상향 돌파**: 매수 신호.
                    - **MACD 선이 Signal 선을 하향 돌파**: 매도 신호.
                """
            )

        # 캔들스틱 차트 표시 (옵션)
        if show_candlestick:
            st.subheader(f"[{stock_name}] 캔들스틱 차트")
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='red', decreasing_line_color='blue'
            )])
            fig_candle.update_layout(
                xaxis_title="기간",
                yaxis_title="가격(원)",
                xaxis_rangeslider_visible=False,
                title=f"{stock_name} 캔들스틱 차트"
            )
            st.plotly_chart(fig_candle, use_container_width=True)

            st.markdown(
                """
                ### 🕯️ 캔들스틱 차트
                - **캔들스틱 차트**는 주식의 시가, 고가, 저가, 종가를 시각적으로 표현한 그래프입니다.
                - **빨간색 캔들**: 종가가 시가보다 상승했음을 나타냅니다.
                - **파란색 캔들**: 종가가 시가보다 하락했음을 나타냅니다.
                - **윗 그림자와 아랫 그림자**: 주식의 최고가와 최저가를 보여줍니다.
                - **패턴 인식**: 다양한 캔들 패턴을 통해 시장 심리를 파악할 수 있습니다.
                """
            )

        # CSV 및 엑셀 파일 다운로드
        csv_data = df.to_csv().encode('utf-8')
        excel_data = BytesIO()
        df.to_excel(excel_data, index=True)
        excel_data.seek(0)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 CSV 파일 다운로드", csv_data, file_name='stock_data.csv', mime='text/csv')
        with col2:
            st.download_button("📥 엑셀 파일 다운로드", excel_data, file_name='stock_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        st.markdown(
            """
            ---
            **📌 참고 사항**
            - **기술적 분석 지표**는 과거의 데이터를 기반으로 미래의 가격 움직임을 예측하려는 시도입니다. 
            - **RSI**, **MACD**, **Bollinger Bands** 등은 각각 다른 방식으로 시장의 과매수/과매도, 추세 변화 등을 파악하는 데 도움을 줍니다.
            - 이러한 지표들은 단독으로 사용하기보다는 여러 지표를 종합적으로 고려하여 투자 결정을 내리는 것이 좋습니다.
            """
        )
