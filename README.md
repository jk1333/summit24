# summit24
Gemini 가 처음이신 분들은 아래 실습을 먼저 진행하시는걸 추천드립니다.\
https://github.com/jk1333/prompt_tester

각 데모별 Main.py의 BUCKET_ROOT 를 본인의 GCS 버킷 이름으로 수정후 아래의 명령으로 Local 에서 실행 가능합니다.\
python -m streamlit run Main.py

Gemini 1.5 Pro의 Default quota는 5 QPM 입니다. 데모 동작에 부족할 수 있으니 100 QPM 으로 증설하거나\
코드 내 THREAD_SIZE 값을 현재 8에서 1로 바꾸어 동작 속도를 늦추면 됩니다.

혹은 Google Cloud 에 배포하기 위해서는 아래의 커맨드를 참고하여 Cloud Run 에 배포하면 됩니다.

gcloud beta run deploy travel --source . --region=asia-northeast1 --allow-unauthenticated --execution-environment=gen2 --timeout=3600 --cpu-boost --no-cpu-throttling --session-affinity --cpu=8 --memory=16Gi --add-volume='name=cache,type=cloud-storage,bucket=TEMP_BUCKET' --add-volume-mount='volume=cache,mount-path=/cache'
