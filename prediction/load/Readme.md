- 입력데이터 필드   
  pg_load_metering table: u_grp_id, metring_value, data_type, tmp 

- 서버에서 u_grp_id에 해당되는 AI 모델 실행 

- 입력데이터 POST 요청(125.131.88.57:8876/predict)   
{  
  "u_grp_id":"UG0000000003",   
  "day_class":0,   
  "meter_value": [10, 20, 30, 40],  
  "tmp": [31, 32, 31, 30]  
}  
- 예측 결과 데이터 응답  
{   
    "predict": "{"0":37.5099807247,"1":37.5099807247,"2":37.5099807247,"3":37.5099807247}"   
} 
