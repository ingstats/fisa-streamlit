import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression

# 페이지 설정
st.set_page_config(page_title="주가 상승 예측", layout="wide")
st.title("📈 주가 상승 여부 예측 (Logistic Regression)")

st.markdown(
    """
    날짜 정보를 활용하여 미래 주가의 상승 여부를 예측하는 단순한 로지스틱 회귀 모델을 구현했습니다.
    아래에 회사명, 기간, 예측 일수를 입력하면 해당 날짜 이후 주가 상승 가능성을 예측합니다.
    """
)

# 사이드바 입력 항목들
with st.sidebar:
    st.header("설정")
    stock_name = st.text_input('회사 이름', value="삼성전자").upper()
    st.caption("예: 삼성전자, 현대자동차 등")
    
    start_date = st.date_input('시작일', value=datetime.date(2019, 1, 1))
    end_date = st.date_input('종료일', value=datetime.date(2023, 12, 31))
    predict_days = st.number_input('예측 일수', min_value=1, max_value=365, value=30)
    
    if st.button("예측 실행"):
        clicked = True
    else:
        clicked = False

if clicked:
    # 회사명으로 티커 가져오기
    try:
        ticker_list = fdr.StockListing('KRX')
    except Exception as e:
        st.error(f"주식 목록을 불러오는데 실패했습니다: {e}")
        st.stop()

    symbol_row = ticker_list[ticker_list['Name'].str.upper() == stock_name]
    if symbol_row.empty:
        st.error("존재하지 않는 회사명입니다.")
        st.stop()
    else:
        if 'Code' in symbol_row.columns:
            symbol = symbol_row.iloc[0]['Code']
        else:
            st.error("티커 정보를 찾을 수 없습니다.")
            st.stop()

    # 주가 데이터 가져오기
    try:
        df = fdr.DataReader(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"주가 데이터를 불러오는데 실패했습니다: {e}")
        st.stop()

    if df.empty:
        st.error("지정한 기간에 주가 데이터가 없습니다.")
        st.stop()

    # 데이터 준비: 날짜를 숫자로 변환하고 레이블 생성
    df = df.reset_index()
    df['Date_ordinal'] = df['Date'].map(datetime.datetime.toordinal)
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Up'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    df = df.dropna(subset=['Up'])

    X = np.array(df['Date_ordinal']).reshape(-1, 1)
    y = df['Up'].values

    # Logistic Regression 모델 학습
    logreg = LogisticRegression()
    logreg.fit(X, y)

    # 예측할 날짜 생성: 현재 마지막 날짜에서 predict_days 후 날짜
    last_date = df['Date'].max()
    target_date = last_date + datetime.timedelta(days=predict_days)
    target_ordinal = np.array([[target_date.toordinal()]])

    # 예측 수행: 주가 상승 여부와 확률 계산
    prediction = logreg.predict(target_ordinal)[0]
    probability = logreg.predict_proba(target_ordinal)[0][1]  # 상승할 확률

    # 결과 출력 (사용자 친화적 UI)
    st.subheader("🔮 예측 결과")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="예측 날짜", value=str(target_date.date()))
        st.metric(label="상승 확률", value=f"{probability*100:.2f}%")
    with col2:
        if prediction == 1:
            st.success(f"{target_date.date()} 이후 주가는 상승할 가능성이 큽니다.")
        else:
            st.warning(f"{target_date.date()} 이후 주가는 하락할 가능성이 큽니다.")

