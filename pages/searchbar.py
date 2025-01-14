import streamlit as st
import pandas as pd

# 텍스트 입력하는 검색창 하나
# 검색어 일부가 들어가면 해당 이미지 출력

df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)

ani_list = ['짱구는못말려', '몬스터','릭앤모티']
img_list = ['https://i.imgur.com/t2ewhfH.png', 
            'https://i.imgur.com/ECROFMC.png', 
            'https://i.imgur.com/MDKQoDc.jpg']

st.sidebar.header("찾을 애니 이름을 입력하세요")
stock_name = st.sidebar.text_input('애니 이름', value="짱구").upper()
clicked = st.sidebar.button("조회")