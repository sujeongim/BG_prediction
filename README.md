# BloodGlucose-Prediction
Code for Blood Glucose Level Prediction


## 1. Task : BG Prediction
- ML 혹은 DL 모델을 사용해서 30분 (6 point 이후)를 예측하는 regression model 개발
- Input Data : { Xt-7, Xt-6, Xt-5, Xt-4, Xt-3, Xt-2, Xt-1, Xt } -> Output value : Xt+6


## 2. Data : T1DMS Virtual Patients BG Data
- BG level of 20 Virtual Patients 
- 14 patients : Train Set / 6 patients : Test Set
- 데이터 1 point 당 5분, 5분마다 data point가 찍힌 것
- (연구실 내부 데이터로 공유는 어렵습니다)

## 3. Model 
- MLP
- LSTM
- Support Vector Regression
- Random Forest Regression
- K-Nearest Neighbors

## 4. Metrics
- RMSE(Root Mean Squared Error)
- MAPE(Mean Absolute Percentage Error)
- Time Gain 

## 5. Data Smoothing
- Moving Average
<img width="1162" alt="스크린샷 2022-06-28 오후 4 34 28" src="https://user-images.githubusercontent.com/50793789/176120739-d7624f8c-6ca9-4394-9d0c-fc0941b45320.png">

## 6. Result
<img width="1147" alt="스크린샷 2022-06-28 오후 4 32 36" src="https://user-images.githubusercontent.com/50793789/176120346-478b9ab6-494c-4beb-b4bb-b267cf295813.png">





