# **Kaggle : Airbnb New User Bookings**

<https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings>

## **Project Summary**

•         Purpose of Project : Airbnb 신규 유저의 첫 번째 숙박 예약 국가를 예측하는 모델을 만든다. 각 머신러닝 알고리즘에 대해 병렬적으로 모델을 만들어보고, Ensemble과 Stacking 기법을 활용하여 본다.

•          Evaluation Metric : 최종 test set에 대한 성능 평가는 NDCG@5로 이루어진다.

•          Best Model : Stacking

•          Score : 0.87014

## **Data Exploration** 

### **Visualizing Data**

**train user 클래스 분포**

![img](https://lh5.googleusercontent.com/RQoC1bR51fPcnZ9slYM7Wprx4arsICb2EoGhDUWkRUYW-oGm5kPSE6_P_E09kJdFnCqrcK87wqUpboLqyw-kM5BEaDFbjJbJCPgbwNSWf4_Mt5qi5gbZ1kUcxtFKVFlCJ2S_iw)

 Training set의 클래스 분포를 확인하면 Class imbalance가 심한 과제임을 알 수 있다. Loss function에서 각 class별로 weight가 빈도수에 반비례하도록 실험을 해보았지만  PT와 NDF의 경우 300배 이상 차이나기 때문에 결과값에 왜곡이 많이 생겨 이를 채택할 수 없었다. 이후의 hyperparameter 조정을 위한 Subsampling for Validation 항목에서는 어느정도 클래스 불균형 문제를 감안하여 subsampling을 하였다.

**Date Account Created/First Active**

![img](https://lh4.googleusercontent.com/fEZNcg2WVu9QAcjudFGNmFAK9uU2S4Gd1b3gRVxZHu9o5-80sD_-pRvGGtJX6ylmECXHoTOy4SrU_OTw4R5oEJ4SysoTp6W_XUDWxTcdzArvyrTl--ydFPBwt6wpnr2gTEXwFQ)

​         <Date Account Created>					<Date First Active>

에어비앤비가 유명해지기 시작한 2014년을 기준으로 가입자 수가 증가한 것을 확인할 수 있다.

**2014년 전후 여행지 비교**

![img](https://lh3.googleusercontent.com/bLYkfxD_1-EcWSVL5xPpuL8jb8UY7GIPUgoti5biKkkAFx3i9M30rsUtIwJm61LXiSCa4QpxEDzFuQ6QJjhGZ2Wxte3q6H1s9Mlijjm9EfUiGqOD3k86gho66ydvYcM9U_Dz6A)

 2014년 이후 에어비앤비가 유명세를 타면서 단순 가격 비교 차원에서 가입을 하는 경우가 늘어 NDF가 증가했다고 생각해 볼 수 있다.

**Holiday에 따른 여행지 비교**

![img](https://lh4.googleusercontent.com/7hJ_kkTgUf5P4MUDiXqCCQIUx1-R_kQA-wILp4gR3QnQuJj1nQiThVpkNI5a7toorLxW-aWNLt6P8Lo69QVCMX9uoJPbN0Dn2ZA8koSHeF-f0zTXP2FpOA481_41TiUmm89aSg)![img](https://lh3.googleusercontent.com/rsdi7MSe26MHX2sDn29P9ztfoyYznVJ3Ee1z3dxocGJac3wBygYS6YmRPJ7btl-lrP0FTlv4o0pIXaFSgQ27MrmK2anWiHeGA3faKW1U70anvzQYqG2fAXo4KyVcBUkTSJ8ZVg)

왼쪽은 전체 Label을, 오른쪽은 NDF, US를 제거하고 플로팅한 결과이다. Holiday 시즌이 비시즌일 때보다 NDF와 US를 제외한 국가들로 여행하는 비율이 높다. Holiday에 더 거리가 먼 국가로 여행하는 것이라고 해석 가능하다.

#### [Checking our Hypothesis](https://github.com/sujinnaljin/airbnb-new-booking/blob/master/project_proposal.pdf)

**가설 1**: 유저가 선호하는 언어와 첫 여행지의 언어가 유사할 가능성이 높다.

![img](https://lh5.googleusercontent.com/sLS9IqZC8TToYzXMcjwkkG7QyEgbqld3Iz-GzVEIfZV48eT1ctOpW4OAJ3haAtObJ5azKJ3QvwDl8UgIt9_nikYWeXGnncVZrDQKKC4949SMK47qoFwVViWZkciKsQ2P2PWfWg)

Index를 여행지로, column을 사용자의 선호 언어로 지정하고 표로 정리해봤을 때, 특정 언어의 사용자는 해당 국가에 여행할 확률이 다른 국가를 여행할 확률보다 더 높음을 확인할 수 있었다. 

![img](https://lh5.googleusercontent.com/FAYGB28IQtVSmohEzapIYpo8L-C9ll4Kva9exntqDTpqoEL45xsa6f-dG30e-Sb6ElyoGcx8wncLSw3Fx4c3dIH-IS1DlAqxZ9CRx8iWV4lV-t_lR5CZ-LGHXG4Zs1T7jQEIPg)![img](https://lh6.googleusercontent.com/P1GRKVlzs7c_F2d75oa_A3HHVIp3qWjE8xWUl-xu0RrJNkda-WMPnvTNC-ww0F4LFJJmD36-wWXlz-fQgu3S2lPUQbMMRtlIDo3ZbHY6ldb3gVjWzmkcnFDEL_4R0QNr3bA7Yw)

![img](https://lh4.googleusercontent.com/G2qU82rQ7XbUV6NDKN3A4on8CMrVhF4H9T_JFQMFcutbHwTohvmvbK9KHJYwktMGijLydjAQlikQRYrMqq8sBbJPuXGW0B_rxBuBBC4z_31Ayn7zNKDxr7JBSPcwbWD2CR8xlQ)![img](https://lh6.googleusercontent.com/nL-qUPQKLcVQSb0f8tEk354T4k_WvgxyQ-xcBH7doAFkfiKyh5T9VvHm65dhmUFTRet14mSKEFxXnYPpDlUIw5RsppKDT1lazHKCc5yMVIvAJt4K1SCCzzeZNI6rpNm4pClR_w)

대표적으로 영어, 프랑스어, 스페인어, 독일어 사용자를 살펴보면, 영어 사용자는 영어권 국가로 갈 확률이 75.1%로 다른 언어 사용자보다 높았고, 프랑스어 사용자는 프랑스로 갈 확률이 17.3%, 스페인어 사용자는 스페인으로 갈 확률이 7.27%, 독일어는 8.50%로 모두 각 언어별 그래프에서 2-3위에 해당하는 높은 비중을 차지하고 있음을 알 수 있다.

**가설 2,3** : 접속 기기 종류와 세션 접속 횟수, 총 세션 접속 시간이 여행지에 영향을 미칠 것이다. 모바일 사용자이며, 접속 횟수가 낮고 총 세션 접속 시간이 짧으면 신경을 덜 쓴다는 뜻이므로 자국인 미국으로 여행을 갈 것이고, 그 반대라면 미국과 거리가 먼 이국적인 나라로 갈 것이라고 가정하였다.

session.csv 데이터를 통해 각 유저가 세션에서 총 얼마나 시간을 보냈는지 구하고, 몇 개의 세션을 열었는지, 그리고 유저가 이용한 각 기기별 이용률을 구하였다. 트레인셋과 합치는 과정에서 left-join merge로 진행하였고, 합친 다음 session 데이터가 있는 유저에 한해서 시각화를 진행하였다.

유저의 총 세션 시간(total_elapsed_time)에 대해서:

![img](https://lh4.googleusercontent.com/yBhpXdwKCyNGkbCJye1DMc4EoIEiOXUcSbYZG9TZAArAoTacxxc8wuJtkY4MN0uU4OTCapqOq1aqi4ves3sNSGTpX9F_iQpMYLtfN_g2i8UxxxhTVlfzgqYfvBj3cWTWVZY4UA)

평균과 중간값을 고려하여 200,000 이하의 세션 타임 유저들을 짧게 이용한 유저(Short Users)라고 임의로 지정하였다.

![img](https://lh6.googleusercontent.com/Uun7p14i4X-xUxBAfsGyVYqbx8wLalmJ7YdXldJw1-bSWS_I3Lklgepg-2yV40VnGGb3HWLwgTp0o2_UcRIy7PAxD0KMU_cHXbA-Fs1ukt_h2IiqAnMCytCjH_GZ0JOElSi1Fw)![img](https://lh3.googleusercontent.com/Eb4MorXgAY-hlaeIFJNZOp8u6_wyfm_833Bqb6QVqa43TEU0rpQReHNDs4Q6NlNOQTB7pMk2ZoaL2YCh7tJDf6wbfymWrHPtL5_sYdJd57c6psw6319Rj1L951aTSlCuLpENnQ)

시각화를 진행한 결과, 총 세션 시간이 짧은 유저들이 그렇지 않은 유저들보다 더 미국에 많이 가고 유럽에 덜 가는 경향이 있는 것으로 관측되어 가설과 일관된 결과를 보여준다는 것을 알 수 있었다.

유저의 모바일 기기 사용률에 대해서(mobile_total)에 대해서:

![img](https://lh3.googleusercontent.com/mGOsZ4qsj4eMjkBKkhXfmCrUw61mJvD3tUwhnSsI7QL69QJgjaFljkoeaZIUW5ewlcLu_0sG99RVkktiolM9qoba9wOnhUusHn-UdN-CMisDqTBs8IBAPDfAq8xNqul300VP4w)

각 기기 중 모바일 사용률만 합친 결과, 극단적으로 모바일을 사용하는 사람과 그렇지 않은 사람이 관측되었고, 나머지 구간은 골고루 사용자가 분포하고 있음을 알 수 있었다. 

따라서 0.0-0.1 구간을 not_frequent_users로 구분하고, 0.9-1.0구간을 frequent_users로 구분하고 임의로 시각화를 진행하였다.

![img](https://lh6.googleusercontent.com/hSDN_CBoGI4R2m1oLmaJga6wzcEYPnwa_pE0aZC4QR1lSNdmzJRS1ET758aOz0GcqMLru487rNzrpENc2GVTfirFO7FL2U5beT6C6FHof9nwdYvi6dXHw4rtsQ9ylOEh5fBpPw)![img](https://lh4.googleusercontent.com/isciVy5kvbgk_9eDQw-AM2RknDPbIVhuazohBSt_t1JUyNFEf6oWZAySfU7IkWlCiahwyHm1lvWy0y8vmJBy-Z0kDgg_MyspMvBZNcKFkvVp3OhALhf4pFjDmY9y5MxlozrhCg)

시각화를 진행한 결과, 모바일 기기를 많이 사용하는 사람이 상대적으로 미국에 많이 가고, 그렇지 않은 사람은 미국과 멀리 떨어진 유럽이나 오스트레일리아에 상대적으로 많이 가는 것을 알 수 있어 어느정도 가설과 일관된 결과를 보여준다고 할 수 있다.

유저의 세션 접속 횟수에 대해서(sess_count)에 대해서:

![img](https://lh5.googleusercontent.com/4YYX2htT40_diWtufBj-yfDdY1bHQpVd9BnKfluS6VPy2fovEoJ-bEnQV4ANO1zyLH7lJl_58hEjejOFgdGBB0gExspHWHkRwRXRG208BKE2mbCxJKaBRnMDK-JcR2zA10apeA)

평균과 중간값을 고려하여 100 이상의 세션 접속 횟수를 가지고 있는 유저들을 (Frequent Session Access Users)라고 임의로 지정하였다.

![img](https://lh6.googleusercontent.com/8W0APhiT8ZbY8pfb0H1EK2dF5Br2RRpZoPGknhUUIrgSCkPb_JFTPIdadqbp8-ExtipgKRIFRmp8PBOrM-DnuHWxL5PF9zKX3Cqtc8D3Z4_YDDkgAoi4FGle8lGgfheL5YUPLw)![img](https://lh5.googleusercontent.com/HX06POba88CABSMX4CBC-bW_xMft6NX-OfCuxeIe48aAQ5zwJrgMSSylJx-l5Dou5PFTcMxgvAU5-TJuAkraG1el3SD8xvDT222cQ-QFIXlOA3Fk8Ajs8_u2b5Ry2TTuRmxV_Q)

시각화를 진행한 결과, 세션 접속 횟수가 높은 사람이 상대적으로 미국에 적게 가고 유럽에 많이 가며, 세션 접속 횟수가 적은 사람이 상대적으로 미국에 많이 가는 것을 관찰할 수 있었고, 이는 가설과 일관된 결과였다. 

**가설 4**: 가입한 방법(signup_method)와 마케팅 채널(affiliate_provider)이 facebook이거나 접속 기기가 모바일, 그리고 처음 이용한 마케팅 방법(first_affiliate_tracked)이 omg일 경우 SNS에 영향을 많이 받는 사람일 가능성이 높으므로 관광지가 유명해 사진 찍기 좋은 영국, 프랑스, 이탈리아, 스페인이 첫 여행지가 될 가능성이 높다.

![img](https://lh6.googleusercontent.com/zKF4TJ3R2yKli2KUTUolCTUGA4U4RJx5GV82oYMNTPIbLI7bqR0J7sE2xlRjiVCO-GgI5H8RQ2usFsR_eqd5KTRtvpEaMavIqCJx8laJI2gCFOykF9Uz2oDgS0cH1BM3Lnc43w)

![img](https://lh3.googleusercontent.com/i4SQ3AeoWTEoaTedAvPPnA1C_JxtevdKjLOvVvAuO_77ptpXpAOTP3Zx5SFj_fSJycNM-tUj94MOISiyOYLtyGXW1prlER7g2BV-kyDHZfM2vEvyvPp1q7DpckP2K2b4kJQhPw)

스페인, 프랑스, 영국, 이탈리아의 페이스북 활용 비중이 높을 것이라고 생각했지만, 스페인, 프랑스, 영국, 이탈리아와 타 국가간 signup_method 항목의 차이는 발견되지 않았다.

![img](https://lh6.googleusercontent.com/pIw3VG2H3610i4Re8enmD4G1ouR7xqYWQKypPseHQ0ImpNCrVXRtrOMZPnU1iSEXMJdiMlfbhwch_8rFQsK-g2C7yEc3z2MRjPmJjFPL2bTEtjjeUhL1AWZq0eE9W-2dPPOvWg)

![img](https://lh3.googleusercontent.com/vqDP6c6OkJomGLr-t1qGWiICtDowSxlRK7XVsph-hrKbK0MMuxLC0qwXQEAYAnFK6HMbKz-U3bO0CR9iULMxLY4IPXyvLZGgL1fpw0EhinQa1uCJCZEunNAqSGs0T3vhV_62ew)

스페인, 프랑스, 영국, 이탈리아의 SNS 관련 사이트 활용 비중이 높을 것이라고 생각했지만, 스페인, 프랑스, 영국, 이탈리아와 타 국가간 affiliate_provider 항목의 차이는 발견되지 않았다.

![img](https://lh4.googleusercontent.com/cRMsZcl1MLOgX2fH46oqmJKKgpF33JMKzdCsU5bUoAeLUei9FOb7rkAHCCfsQ8_KeAlRpryXJGnizd2mu6cJ0vc-CzDXQmUs9U0MpYybmR4tKpL9fCnuru8h6lTmSmzy4ASm_Q)

![img](https://lh6.googleusercontent.com/DQ6PM4wlvO1-_iF9qBWQtukonJzzmuU1VKztCJNyECrXSTDixH05YDxwU9CLW5ECnt4lG3fyd_ieJByB0Xdd3mtK4ajrkhBJacn4e-TkRsPkU2ICweCDNVj_qtdWSM7px5u01w)

스페인, 프랑스, 영국, 이탈리아의 omg 활용 비중이 높을 것이라고 생각했지만, 스페인, 프랑스, 영국, 이탈리아와 타 국가간 first_affiliate_tracked 항목의 차이는 발견되지 않았다.

![img](https://lh3.googleusercontent.com/sWyh271HvQyDyvLNX8j4zlKMLv0Cp0Q-7IdR_QnMbu2VQviJILg8TP-Kt7oaSrot6JN7ExaJTTxn99wgJKUieerFH5cjRISb_c0_PJOuvoLOSzoN2IICjmf7nRD7j2iX4IWiUQ)![img](https://lh6.googleusercontent.com/qkX5pf10THYjnsMwqEZZtN8qFFVTAH3Sr0NZg7ODnw602TuoI75nF2PNru02jaBm4cppDYqK2y8hgjhVfJP2RzSLuM_SoiptXFqLba9lnbYyL8VyDck32e94B4qe8h6eiBOqNg)

스페인, 프랑스, 영국, 이탈리아가 처음 접속한 기기(first_device_type)는 모바일이 높을 것이라고 생각했지만, 스페인, 프랑스, 영국, 이탈리아와 타 국가간 first_device_type 항목의 차이는 발견되지 않았다. 다만, 여행지를 정하지 않은 경우의 유저는 Mac Desktop, Windows Desktop 등의 데스크탑 비중이 더 낮고 iPhone 등 모바일 기기 사용이 더 높은 것으로 드러났다.

**가설 5:** 유저의 first_device_type 별로 평균 연령을 조사해보았을 때, 상대적으로 모바일 기기 사용자들이 연령층이 낮을 것이다.

![img](https://lh5.googleusercontent.com/qifKxK-iGR45T8AHQ5vRbe_oSVAnIkj441bzngMF_MvPbapZzDfpopa7n_14Gmc5iB339WmKiWmR__swioeqTzl_UonWS0iIcVKBqVbSIFzketkASLiAZziJYeavZIHQqA0WNg)

 ![img](https://lh4.googleusercontent.com/s1vWWR0fXZOITU35xkiNvebSWlMQtq_W2Fp-HuuRcGEohkBSSyshFNSDITT66oAWuyks6Vwd6dx10VBGdks1ko-FGnTD7N7LVRRyh7p_hjrgmyXJJWZuDRdVoZV797XHPgFR5Q)

전체적으로 모바일 폰<데스크탑<타블렛 순으로 평균 연령이 나타났으며, 모바일 폰의 경우 연령대가 젊은 층에 몰려있지만, 데스크탑과 타블렛은 상대적으로 연령대 스펙트럼이 넓고 연령대가 더 높다는 것을 확인할 수 있었다.

![img](https://lh6.googleusercontent.com/OyodClj0jiohPR2wZLgCnwoONHj7Vc69kxASVgPVU7gaj55LtoKGrP40hW1IxGojd-5v0W5knHUSd8Px3aYIRCuWk0Gw4opmPdgybhgkZNCUdUouU6fS7NkJBchOYcYNndR_mQ)![img](https://lh3.googleusercontent.com/fjzA_UX2Ugi_7nkdvY6S1umw9jowm-u2xa8bT60-i7ASts0AXqPfb3gvcZOiOO4yS5KmmuJDBP_V67v5jPBQUOQEtZF3WURBi9ibt4w6TgJfULBU6wKPtrEkknJ4I95mHKViKQ)

 

## **Feature Engineering**

### **Convert Date Time Feature**

1. date_account_created(dac) -> year, month, day: date_account_created(계정 생성일) feature로부터 계정 생성 연도, 월, 일 feature을 생성하였다. month같은 경우 계절, day같은 경우 월말,월초와 같은 feature를 나타낼 수 있다.

1. timestamp_first_active(tfa) -> year, month, day, time: timestamp_first_active(첫 활동 시간) feature로부터 첫 활동 연도, 월, 일, 시간 feature을 생성하였다.

### **Fill in Missing Data**

#### Gender

##### a ) Gender 구성

![img](https://lh3.googleusercontent.com/B1aJC7rwlg6HGtsx3tI7s5nOfQT-Stvp-2t87BZ1gCYZ3UIZOnT-Ht63cgCzaTNvjxlWRweymXS5l5z7ZV6SL_6fPwb0HgIiyDbH1VpoO9wIlMVdMipG4Jpy4GiQnXi5oGhhfg)

​                            <train과 test set의 성별 구성>

=> 사용자의 능동적 선택의 결과인 other은 그대로, 미기입의 결과인 -unknown-은 male, female 중 랜덤하게 채워넣었다.  

##### b) Gender에 따른 여행지

![img](https://lh6.googleusercontent.com/WWJ-RhlmWxZHAwqKDOXKXapZ3Vwanm8Gk0EOQPFkD-ciSQxHedS-u3O8bD5MEPh4N1dqMROXjzWmIVxqJzEgQC-JPpNmc3b0oZAnVP6LvgsnKT0f02DCVMFUD7HMN4nIC66lZg)

 ####  Age

 ##### a) 분포

![img](https://lh3.googleusercontent.com/iU25ozZd6RY8L7xnf9i791mDc-LJCcwVXd5uYwweQq0owWYT-wkC_Hi4I0RP5UPCCAACzFsuZNdjivvVTGwc4lVql4POhhVzPUYWxOz0eu1antm2KfkjCrWa53YzcWI1hvmSSg)

##### b) Missing Value : train user set의 age의 46%는 Missing Value

##### c) Outlier

​           			         ![img](https://lh3.googleusercontent.com/fBfo00iwVqwhEmZPEZQQm-OyC6z8J_evIi-Oi8LHG_5Iz7RnZZwI7laiZPmFArdyddBeUHEHzuqYWhyGgiBz9DYkRd7ZBVTH5v5SbfayJyPnP63z_We08TXBEnT7YN_Wm7gBHQ)

 			             <1900살이상>                   <18살이하>							,

   1920-2000의 경우 본인의 생년을 입력한 것으로 추측된다. 따라서 date account created(2010-2014년)의 값을 빼주어 정확한 나이 값으로 대체한다.

 2000년 이상인 경우 date account created을 빼주어도 최대 10살이 된다. 미국의 경우 11살부터 중학생에 해당하게 되는데 초등학생이 에어비앤비를 예약하는 것은 합리적이지 않아 Missing Value로 처리한다.

 마찬가지로 18살 이하 중에서 10살 이하는 Missing Value로 처리한다.

##### d) Filling in Missing Value

​      우선 Gender를 Male과 Female 둘 중 Random Value, 그리고 Age를 제거한 채로 Random Forest를 적용해보았다. 그리고 Feature Importance를 구한 결과 다음과 같이 age feature의 importance가 가장 높게 나왔다. 따라서 age feature를 단순히 mean value로 채우기 보다 다른 feature들을 활용해 예측해보기로 결정했다.

​                               ![img](https://lh5.googleusercontent.com/T7fhZaAknQKwOxLhEJ-mPuB3M7_f6oT-VsA0baNcXznGj5Fu75ZuzT3idBIK1cyDWm32Wq-RENDF28K6vbGkVkgca4l2JExhP3BFTWrk68gqm5BzL_J7BZOz4Xl-41UGVEns7g)

<Random Forest Feature Importance>

##### e) Predicting Age

 Age를 예측하기 위해 Xgboost, LGBM, CatBoost, Deep Neural Network를 활용했다. Deep Neural Network는 쉽게 overfitting되었고 Tree 계열의 경우 예측력이 현저히 떨어졌다.

​                       ![img](https://lh5.googleusercontent.com/DRJBzFmp97Io_B-cFn7NT_V7a1M-GsNAxoWl1LLJtLNyQ5ykzmJjtwdfgFw33h7FFPIKVClblIuMPj5Bd6sSIzkQPYVB36a-hS4HEHDwJdA9ZAUm04SdagArXKTPblF0MAdR3A)  ![img](https://lh6.googleusercontent.com/VGuU5K18tuDtw3gdCvhmGze6Ndc8_2IyuFw6IWFcJfr7SqFR54yNJ4A9ciZfXxpd8FEA3UrCTsj-4NPC3v9sXeWMMUaaIiepUxFiz9oe181Fa0HH9tFeQfqh3eV2Cjn-dMQCUg)                                       

​                ![img](https://lh4.googleusercontent.com/_zq6-6bQ1PB46TV8C_3qs6BLJYPTNtZlsCzhzykDWjJ9khlO3BiyTbUgrk_1fGGEBNVMx2OG-LWUJt5nfse_FhCDRa81BxVOJtY5x3fxSjjNJj0L8X5sMqGTHHKjxkkvFaDKOA) ![img](https://lh3.googleusercontent.com/QSI-9hLiiXnAJiraXTvqpla4V1zad2bvinr1gOP7w_4BB-ihyT-XKukiBIIr0ZkdZuakm4eCweudhlANNhiRoPFiQL99b_NewrUKE27eMNuwms3QzNB04kXjutfHrmL6ZxRAaw)            

 

 따라서 Variance가 커지는 모델을 선택하기보다는 조금 더 robust한 방법인 Mean Value로 Age의 Missing Value를 채워넣기로 결정하였다.

 

### Drop meaningless Feature

 Date First Booking feature는 train set에서 ‘NDF’ 클래스에 해당하는 약 58%가 null value이다. Date First Booking은 첫 예약을 마쳐야만 생겨나는 사후적인 정보이기 때문에 prediction에 활용하는 것이 논리적으로 맞지 않아 이 feature는 활용하지 않기로 결정하였다.

### **Create additional Features**

1. date_account_created(dac) feature -> season, weekday, pred_diff: date account created feature의 정보를 이용하여 계정 생성일의 계절, 요일, 예측 연도(기준 연도)로부터의 연도 차, 즉 예측 당시 기준으로 계정을 생성한지 몇 년이 경과하였는가를 나타내는 feature을 생성하였다.

1. timestamp_first_active(tfa) -> season, weekday, pred_diff, time range(trange): timestamp first active feature의 정보를 이용하여 첫 활동 시간의 계절, 요일, 예측 연도(기준 연도)로부터의 연도 차, 즉 예측 당시 기준으로 첫 활동을 한지 몇 년이 경과하였는가, 그리고 첫 활동 시간의 시간대를 나타내는 feature을 생성하였다.

### **One Hot Encoding**

대상 feature: 'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_weekday', 'dac_season', 'tfa_season', 'tfa_weekday', 'tfa_trange', 'dac_year', 'tfa_year'

### **Normalization using Z-score**

대상 feature: 'age', 'dac_pred_diff', 'tfa_pred_diff', 'dac_month', 'dac_day', 'tfa_month', 'tfa_day', 'tfa_time'

 

## **Model selection**

### **Subsampling for Validation**

Validation 과정을 거칠 때 기존 training set에 training을 매번 새로 하기에는 training set의 크기가 너무 커서, subsample을 만들어 validation 과정상의 training set의 용도로까지 겸하여 사용하였다. 즉 연산비용 문제로 training set의 subsample set이 validation 과정 상의 training set과 validation set 역할을 모두 담당하였다. Subsample set을 활용한 model selection(hyperparameter 선택)으로 선정된 모델들은 마지막으로 전체 training set에 대해 training을 시키고 test set으로 최종 성능을 평가한다.

Subsample을 생성할 때 두 가지 방안이 있었는데, 한 가지는 기존 training set의 클래스 비율을 유지한 subsample을 만드는 것, 다른 한 가지는 minority 클래스가 기존 training set에 비해 oversampling되도록 하고 majority 클래스는 undersampling되도록 하는 것이었다. 후자의 경우 minority 클래스가 test set에서도 무척 적을 것이기 때문에 후자를 이용해서 hyperparameter을 설정하면 전체적인 성능에 오히려 악영향을 끼칠 가능성도 있다고 판단하였지만, 최종적으로 training set에서 majority 클래스의 datapoint가 충분히 많아 hyperparameter에 관계없이 잘 학습할 수 있으리라고 판단하고, datapoint가 적어 학습에 어려움이 있는 minority 클래스에 조금이라도 더 집중하는 hyperparameter을 구성하기로 결정하여 후자의 방안을 선택하였다.

​	subsample은 ‘make_subset.ipynb’를 통해 만들었다. 크기는 datapoint 약 1만 개로 하고자 하였다. 이를 만족하도록 클래스마다 샘플링할 개수를 정해두고 해당 개수만큼 비복원 추출을 하였다. ‘PT’는 datapoint 수가 217개로 전체 추출, ‘AU’은 539개 전체 추출, ‘NL’은 762개 전체 추출, 이 외의 클래스들은 datapoint 수가 1000개 이상인 클래스들로, 모두 1000개를 추출하였다. 결과적으로 10,518개의 datapoint로 이루어진 subsample이 생성되었다.

 

### **HyperParameter Selection for Each Model**

#### **Logistic Regression**

Logistic Regression의 hyperparameter를 찾기 위해 GirdSearch CV에 비해 실행 시간이 적게 들면서도 성능이 현저히 떨어지지 않는  RandomizedSearchCV를 실행하였다. 이를 위해 우선 fitting 에 사용할 parameter grid를 설정하였고 그 값은 다음과 같다.

```python
penalty =  ['l1', 'l2']
C = [0.001,0.01,0.1,1,10,100]
```

가장 최적의 성능을 보이는 parameter의 조합은 다음과 같다.

```python
{'C': 1, 
 'penalty': 'l1'}
```

L2 norm보다는 L1 norm을 이용한 regularization이 선호된 것으로 보아 prediction에 도움이 되지 않는 feature이 많은 것으로 추측된다.

#### **RandomForest**

RandomForest의 hyperparameter를 찾기 위해 GirdSearch CV를 실행하였다. fitting 에 사용할 parameter grid 값은 다음과 같다.

```python
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
```

가장 최적의 성능을 보이는 parameter의 조합은 다음과 같다.
```python
{'bootstrap': True,
 'max_depth': 10,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 5,
 'n_estimators': 200}
```



#### **Xgboost**

Xgboost의 hyperparameter를 찾기 위해 GirdSearch CV를 실행하였다. fitting 에 사용할 parameter grid 값은 다음과 같다.
```python
min_child_weight = [1, 5, 10]
gamma = [0.5, 1, 1.5, 2, 5]
subsample = [0.6, 0.8, 1.0]
colsample_bytree = [0.6, 0.8, 1.0]
max_depth = [3, 4, 5]
```



가장 최적의 성능을 보이는 parameter의 조합은 다음과 같다.
```python
{'colsample_bytree': 0.6,
 'gamma': 5,
 'max_depth': 5,
 'min_child_weight': 5,
 'subsample': 1.0}
```



#### **CatBoost**

CatBoost loss function: 

![img](https://lh5.googleusercontent.com/QMwjdbeJbd0dW1kwNT67adYnlvkxiyJS3r-Mk3sskidxIAie0q0-hNqlHtZENE7Ax8DSkHLYTDzOR-dfnrZ0uvbjCkRPzYIfk7M9Aam8wz5hWC-4Y8jsDRQsxFMBEKe_MY0KRA)

CatBoostClassifier 파라미터에 해당하는 depth의 적합한 값을 찾기 위해 수동으로 테스트를 진행하였다.

--Depth: 5

| Learning Rate  | 0.1          | 0.05         | 0.01         |
| -------------- | ------------ | ------------ | ------------ |
| Best Score     | -2.391290585 | -2.393955152 | -2.422739425 |
| Best Iteration | 95           | 98           | 99           |

--Depth: 10

| Learning Rate  | 0.1          | 0.05         | 0.01         |
| -------------- | ------------ | ------------ | ------------ |
| Best Score     | -2.413476377 | -2.410183727 | -2.436579506 |
| Best Iteration | 32           | 78           | 99           |

따라서 CatBoostClassifier의 parameter은 다음과 같이 설정하였다.

```python
iterations=95
learning_rate=0.1
depth=5
task_type='GPU'
loss_function='MultiClass'
eval_metric='MultiClass'
```



#### **Neural Network**

Neural Network 모델에서 적합한 coefficient, hidden layer, hidden layer 당 노드 수, batch size를 찾기 위해 수동으로 테스트를 진행하였으며 fitting 에 사용한 parameter grid 값은 다음과 같다.
```python
Regularization = [l1(0.01), l1(0.001)]
hidden layer and nodes = [(2, (64, 32)), (2, (128, 64)), (3, (128, 64,32)), (3, (256, 64, 32))]
batch_size=[10,100]
```

-->batch size가 10일 때는 컴퓨팅 파워의 한계로 l1(0.01)만 적용하였다.

성능에 대한 평가는 categorical_crossentropy로 계산 되었다.

--batch size:100

| Hidden Layers and Nodes/Regularization | l1(0.01) | l1(0.001) |
| -------------------------------------- | -------- | --------- |
| 2layer(64,32 nodes)                    | 1.2075   | 1.2704    |
| 2layer(128,64 nodes)                   | 1.2385   | 1.3647    |
| 3layer(128,64,32 nodes)                | 1.2040   | 1.3548    |
| 3layer(256,64,32 nodes)                | 1.2675   | 1.5433    |

--batch size :10

```python
3layer(128,64,32 nodes):1.2045
2layer(64,32 nodes):1.2174 
```



표에서 볼 수 있듯 최적의 성능을 보이는 parameter의 조합은 다음과 같다.
```python
{'Regularization': l1(0.01),
 'Hidden Layers and Nodes/: 3layer(128,64,32 nodes),’batch size’:100}
```



#### **LGBM (Light Gradient Boosting Machine)**

LGBM의 hyperparameter를 찾기 위해 GirdSearch CV를 실행하였다. fitting 에 사용할 parameter grid 값은 다음과 같다.
```python
learning_rate = [0.01, 0.1, 0.5]
max_depth = [5, 8, 10]
num_leaves = [10,15, 50]
```



가장 최적의 성능을 보이는 parameter의 조합은 다음과 같다.

```python
{'num_leaves': 10, 
 'max_depth': 5, 
 'learning_rate': 0.01}
```



#### **Ensemble**

 LGBM, XGBoost, Catboost 위 3 결과를 평균 내어 더 robust한 모델을 만든다.

![img](https://lh3.googleusercontent.com/F5acoicr3DF9ecOChcCfGWedyp4VBoVCPr4coUDWFFRaLDi1ZxG-zbbBaKNKtCAk2sHHg-Vk-cZi2hvpLorlvABJoKgWvYyQHUh5Id51qFV_1nbsrfThBh4gxtPqFNzWfkBj-A)

#### **Stacking**

결과적으로는 Xgboost가 가장 좋은 성능을 보였다. 하지만 컴퓨팅 파워의 한계로 XGboost 대신 GPU연산이 가능하고 연산 속도가 빠른 Catboost Model 2개를 활용해 Stacking을 구성하였다.

1st layer model의 하이퍼파라미터는 기존의 것을 이용하였고 해당 layer에서 나온 각각의 class에 대한 확률값을 피쳐로 사용해 2nd layer의 모델을 학습시켰다. 2nd layer model 후보로는 Random Forest와 Neural Net가 있고 각각에 대한 하이퍼파라미터 튜닝을 실행하고 비교해본다.

--Neural Net : val_loss

--batch size:100 일때

```python
2layer(64,32 nodes) : 1.1874 

2layer(128,64 nodes): 1.1411  -->score: 0.86943

3layer(256,128,64 nodes) : 1.2286
```



--batch size:10일때

```python
2layer(128,64 nodes):1.0986   -->score: 0.87014

2layer(64,32 nodes):  1.0997
```



--Random Forest

![img](https://lh6.googleusercontent.com/uigF7zsegLNzXV7oU-NgoNqOSHNnMtwlXI7hIDVwysYkp8gbsjiilVa8IYr5CFRukl1lbQ7Zjq41ZZrTAXEJB6PXlnJQX35XqQPrqG8fkrjZaeuntB1ECuOwVZf4vQNTmmSsbw)--->Random GridSearchCV

![img](https://lh3.googleusercontent.com/ZZN7XHklfXLyKIg0-OX69Yy5XnNxPCIMajfjdVMD4B_uudf8JOFgW4J6Ox-7p5OPO1APHT_bGji9D9Gnm0FmjNXAJTtB5O385T7whtFQaA_e7fopNFEl_3pryMIH2sKroIDxKA)-->best Parameter

Random Forest score:  0.86282

따라서 2nd layer model은 Neural Network로 구성하였다.

**![img](https://lh3.googleusercontent.com/B58QTEtugbW6YTVG4-DNH5maIf3PIPNHMaq4_b5PXtK7ekqX4iMV-n5VxF-lM0T2fx2YB0JoHDYIaFJ8aRFJ_hnRIPLKVMDk_ftT24Gv5FaGt9wZyBJ93ln7mZfhnT89HXIlbw)**

**<Stacking Structure>**

## **Test results**

Test set에 대해 각 모델의 성능을 측정하였다. test set의 실제 클래스 정보는 주어져 있지 않은 관계로, Kaggle에 submission을 함으로써 얻는 NDCG@5 점수로만 성능을 평가할 수 있다.

### **Base Line -Majority Label: NDF**

1위부터 5위까지를 전부 NDF로 채워서 제출하였다.

**Score :** 0.67908

### **Logistic Regression**

**Score** : 0.85373

### **RandomForest**

**Score** : 0.85356

**Feature Importance**

![img](https://lh4.googleusercontent.com/o162YDtiJOlK6b1kEJt8zKIHGchp9WkgNxZuaBdGv1tg_mTBHLhkSs4805yK6saiLBEqQCpT8HJi-cj7N9rhr4lJFd1eYvFjbcOKlGgIqzIs8r8AKTlbmQG6Gs678Hz1toe6bA)

### **Xgboost**

**Score** : 0.86561

**Feature Importance**

**![img](https://lh5.googleusercontent.com/2h6TgZ02apm77rvRfbnI4WBTGC3mi20CgiuVgX3OaF3cyGCHwaV-IbCek3dDseVtS_FXn6lrFO4poIOdv_DYkeOEri6JDxCN3VIl8lAs7EBr4aosIDbQIPulHJzLrbL5cTbj1g)**

### **CatBoost**

**Score** : 0.86508

**Feature Importance**

**![img](https://lh4.googleusercontent.com/fUeDm5OegQyWqidx5kJaX5EKn8fe-OvS7hI1XLKB1CcVTkkur1yeNnnFPAnrpfuYVWa5ojXpJ6eobEE_s27kUXqH9eyzb1L9IQCi_MF1RWcadWr8MNAXqiT8z8gJtDcP7nPjpg)**

### **Neural Network**

**Score** : 0.85359

### **LGBM (Light Gradient Boosting Machine)**

**Score** : 0.85369

**Feature Importance**

![img](https://lh5.googleusercontent.com/NRYEPyaGViLCOsGRkEpKZsKo09rYbWdpuAyCaN3wBa1o-Cg0KDmx3RXJBJCC2g0mLSYRSzR-d_TYIUv1dgh94a6VU4F-CxuRr5q4JhKEHNlLrtSzZS2r3n_INBYO7J5UmfA5Cw)

따라서 가장 좋은 성능을 보인 단일모델은 Xgboost로 0.86561의 정확도를 기록하였다. 

그 외 모델의 성능 순위는 아래와 같다. 

**CatBoost>Logistic Regression>LGBM>Neural Network>RandomForest**

Xgboost, Catboost, LGBM, Random Forest의 Feature 중요도 상위 10개 feature들을 종합하여 보면 다음과 같다. 다음은 각 feature 별 각 모델에서 상위 10개 중요 feature에 든 횟수이다. 전부 16개의 feature들로, 서로 다른 모델끼리도 대부분의 feature이 겹치는 것으로 보아 안정적으로 중요한 feature들이라고 할 수 있다.

1. age 's importance:  4
2. tfa_pred_diff 's importance:  4
3. signup_method_facebook 's importance:  4
4. tfa_time 's importance:  3
5. dac_pred_diff 's importance:  3
6. signup_method_basic 's importance:  3
7. signup_flow_3 's importance:  3
8. signup_app_Web 's importance:  3
9. dac_month 's importance:  2
10. tfa_month 's importance:  2
11. first_device_type_Mac Desktop 's importance:  2
12. affiliate_channel_content 's importance:  1
13. affiliate_channel_other 's importance:  1
14. first_device_type_Other/Unknown 's importance:  1
15. first_browser_-unknown- 's importance:  1
16. first_browser_Chrome 's importance:  1

 나이는 모든 모델에서 공통적으로 압도적으로 가장 중요한 feature이었다. 그 외에 항상 중요하게 꼽힌 feature로는 첫 활동을 하고 경과한 햇수, 가입 계정이 facebook 계정인지 여부이다.

 다음으로 3개 모델에서 중요하게 꼽힌 feature로는 첫 활동의 시각, 계정 생성을 하고 경과한 햇수, signup_method가 Airbnb에 직접 가입하는 것이었는지 여부, signup_flow 3번의 URL을 통해 Airbnb에 유입되어 가입한 것인지 여부, 가입할 때 웹 애플리케이션을 통했었는지 여부가 있다. signup_flow의 경우 번호로 암호화되어서 실제로 어떤 사이트였는지는 알 수가 없다.

  2개 모델에서 공통적으로 중요했던 feature들은 계정생성 월, 첫 활동 월, 그리고 첫 접속 기기가 Mac Desktop이었는지 여부였다. 마지막으로 한 개 모델에서 중요하게 평가된 feature들은 사용자가 접한 마케팅 수단과 첫 접속 브라우저 종류 및 첫 접속 기기의 종류에 대한 것들이었다.

1. Age 

![img](https://lh5.googleusercontent.com/oH9MdzaJv4RborsCvOAQpODAghM8s5_5dBeCX6fMc0czF0_XS6bv1ARVtTwZxXIWBc1QDBK9UgZKQKJXXuHjrLo7k7RT17VMNiK-FPPrDYijL5PS3Wl8JGFdKSsLrrRuVHfiKQ)

​                                                

위 그래프에서 40대의 경우 NDF일 확률이 높은 것이 나타난다. 자세한 분포를 알아보기 위해 Stacked Bar Chart를 확인한다.

![img](https://lh6.googleusercontent.com/fuWDy5EQMohElxgykoMKOXvI7PQsSbduTMEZmdDjz28iBqSiXmgvg2snrWONS-S1jpc00ZZRTkGwPsyZ3PkJ4O2qj8V7BQNG9h_58M5xW68rsmM9auJ_MADndKq-5P-S68IjUA)

Underage(<18) 와 Fourties에서 NDF 비율이 높은 것을 알 수 있다.	또한 40대의 경우 Australia의 비율이 적은 것을 확인할 수 있다.

![img](https://lh6.googleusercontent.com/9kzeP-WE58NkrCa6Y5-jkIAvtTVDUw08g_nYm7ipgRjPDhM-jvaC9PBcqb-vu5uWOKfQ3CGMkDvo_Id_ddG9MpJMSZeWs1mpsxBUJMWOP1ZHNf2tzHsXO3Bkun8pgsiOELiBVA)

​        <40 대의 여행지 비율>

2) singup_method_facebook

![img](https://lh3.googleusercontent.com/ErvVAX43qHvOTGAqKsXUJuBwWO6Ainblneey-UjNBJyBtjn8MfJaGApZzYgcoRBhkt-XXcPPiaVbp203ISSwsY2OyafgD5PXb5Rr_Xq3Pw9aQ1O7Dm9vNONA3H_9nKeDxeOlKQ)

 DE를 목적지로 하는 경우 facebook을 통한 가입 비율이 다소 높을 뿐 그래프를 통해 통계적 의미는 찾기 힘들다. 

![img](https://lh4.googleusercontent.com/B0onOyxFopg15mFkyMw4szXI2jeS2fPraoru6_t5LOEI88RVPuWhXKQZj7eDaQuzZKpCcn-OcY8_-lXK2Rp7t0XQJmiW0I3Qyn6PkFKeNdLPOOzgh8FJPEbdYs8r5HpVkX1i7w)

 40대의 경우 facebook을 통해 가입하는 비율이 굉장히 적다.

![img](https://lh5.googleusercontent.com/oTUreJRjnIAV-PLRQe1s-q6gvs4NELHVwWo_xFtIjTBGVAAq10ppQF8PguWTZH_jIVkAVNkx8kg53iw4sE0Mtu8SDtPNVGmttUibSfbOO_aUW6yfOCcCbB1AfOQ8dLhgCF67tQ)

facebook을 통해 가입한 40대와 그렇지 않은 40대의 목적지를 비교해보면 facebook을 통해 가입할 경우 US로 향할 확률이 높다.

### **Catboost_2:** 

위에서 서술한 Catboost의 feature importance 상위 10개 feature만을 활용해 test하였다

score: 0.86533

-->전체 데이터 셋을 넣었을 때보다 성능이 향상되었다. 다른 feature들이 주는 noise가 없어졌다고 해석가능하다.

### **LGBM_2 :**

위와 마찬가지로 LGBM의 feature importance 상위 10개 feature만을 활용하였다.

Score: 0.85350

### **Deep Neural Net_2 :**

위와 마찬가지로 DNN의 feature importance 상위 10개 feature만을 활용하였다.

 Score: 0.85359

### **Ensemble(LGBM,XGBOOST,CatBoost 평균):**

score:0.86385

### **Stacking:**

score: 0.87014



## **Further Goals**

1. Session data의 활용: 

   1. user가 실행한 action 개수
   2. user가 실행한 action 가지수
   3. device_type 가지수
   4. user가 사용한 device_type 중 가장 많이 사용한 device type
   5. device_type unique한 종류 중 각각 사용한 횟수
   6. user의 secs_elapsed 총 시간
   7. user의 secs_elapsed 평균 시간

​	(참고 파일: Add_Session_features.ipynb)

2. 연산비용으로 인한 제약 해결: 연산비용의 제약으로 인해 Session과 관련된 feature은 쓰지 않고 validation 과정에서 subsample을 통해서 training을 해야 했다. 또한, Stacking에 XGboost를 활용하지 못하였다. GPU 대여 서비스를 적절히 활용하여 연산비용으로 인한 제약 없이 모델을 만들 수 있을 것이다. 

## **Role**

이세훈 - 기존의 유사한 연구나 프로젝트 조사, Model building & training & refinement

정예람 - Exploratory Data Analysis & Visualization, Model building & training & refinement

조민서 - Data Preprocessing, Model building & training & refinement

강수진 - Exploratory Data Analysis & Visualization, Model building & training & refinement

임한동 - Data Preprocessing, Model building & training & refinement
