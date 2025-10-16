# YOLOv11 JPEG Ensemble 사용 가이드

## 개요

이 구성은 JPEG 바이트를 입력으로 받아 YOLOv11 추론을 수행하는 앙상블 모델입니다.

### 구조
```
yolov11_ensemble
├── preprocessor (JPEG bytes → FP32 tensor)
└── yolov11 (FP32 tensor → detection results)
```

## 디렉토리 구조

```
rbln_backend/
├── preprocessor/           # JPEG 디코딩 및 전처리 모델
│   ├── config.pbtxt       # BYTES 입력, FP32 출력
│   └── 1/
│       └── model.py       # JPEG 디코딩, 리사이즈, 정규화
├── yolov11/               # 실제 추론 모델 (기존)
│   ├── config.pbtxt
│   └── 1/
│       ├── model.py
│       └── yolov11.rbln
└── yolov11_ensemble/      # 앙상블 정의
    └── config.pbtxt       # preprocessor → yolov11
```

## 1. Triton Server 시작

```bash
tritonserver --model-repository=/workspace/kepco/triton/rbln_backend
```

서버가 시작되면 다음 3개의 모델이 로드됩니다:
- `preprocessor`: JPEG 전처리 모델
- `yolov11`: YOLOv11 추론 모델
- `yolov11_ensemble`: 앙상블 모델 (이것을 사용!)

## 2. 테스트 데이터 생성

### 방법 1: 여러 개의 JPEG 이미지 생성 (권장)

```bash
cd /workspace/kepco/triton
python3 create_jpeg_input.py --num-images 10 --output-dir ./perf_data
```

옵션:
- `--num-images`: 생성할 이미지 개수 (기본값: 10)
- `--output-dir`: 출력 디렉토리 (기본값: ./perf_data)
- `--quality`: JPEG 품질 1-100 (기본값: 95)

생성 결과:
```
perf_data/
├── image_0.jpg
├── image_1.jpg
├── ...
├── image_9.jpg
└── input_data.json  # perf_analyzer용 설정 파일
```

### 방법 2: 단일 JPEG 바이너리 생성

```bash
python3 create_jpeg_input.py --single
```

## 3. perf_analyzer로 성능 테스트

### 기본 테스트 (합성 데이터)

```bash
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --shape IMAGE_BYTES:60000 \
  --concurrency-range 1:8:1
```

참고: `--shape IMAGE_BYTES:60000`는 대략적인 JPEG 크기입니다 (실제 크기에 따라 조정)

### 실제 JPEG 데이터 사용 (권장)

```bash
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --input-data=/workspace/kepco/triton/perf_data/input_data.json \
  --measurement-interval=5000 \
  --concurrency-range 1:8:1
```

### Request Rate 기반 테스트

```bash
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --input-data=/workspace/kepco/triton/perf_data/input_data.json \
  --request-rate-range=100:200:10 \
  --max-threads 32
```

### Shared Memory 사용

```bash
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --input-data=/workspace/kepco/triton/perf_data/input_data.json \
  --request-rate-range=100:200:10 \
  --max-threads 32 \
  --shared-memory system
```

## 4. Python 클라이언트 예제

```python
import tritonclient.http as httpclient
import numpy as np
import cv2

# JPEG 이미지 읽기 및 바이트로 변환
image = cv2.imread('test.jpg')
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
success, encoded_image = cv2.imencode('.jpg', image, encode_param)
jpeg_bytes = encoded_image.tobytes()

# Triton 클라이언트 생성
client = httpclient.InferenceServerClient(url="localhost:8000")

# 입력 텐서 생성
input_data = httpclient.InferInput("IMAGE_BYTES", [len(jpeg_bytes)], "UINT8")
input_data.set_data_from_numpy(np.frombuffer(jpeg_bytes, dtype=np.uint8))

# 추론 요청
result = client.infer(model_name="yolov11_ensemble", inputs=[input_data])

# 결과 가져오기
output = result.as_numpy("OUTPUT__0")
print(f"Output shape: {output.shape}")  # (1, 84, 8400)
```

## 5. curl 테스트

```bash
# JPEG 파일을 base64로 인코딩
base64 test_image.jpg > image_base64.txt

# HTTP 요청 (간단한 테스트)
curl -X POST localhost:8000/v2/models/yolov11_ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "IMAGE_BYTES",
      "shape": [60000],
      "datatype": "UINT8",
      "data": [...base64 data...]
    }]
  }'
```

## 성능 최적화 팁

1. **Preprocessor 인스턴스 수 조정**
   - `preprocessor/config.pbtxt`의 `count` 값 조정
   - CPU 코어 수에 맞게 설정 (기본값: 4)

2. **JPEG 압축 품질 조정**
   - 높은 품질 = 큰 파일 크기 = 느린 전송
   - 낮은 품질 = 작은 파일 크기 = 빠른 전송, 낮은 정확도
   - 권장: 85-95

3. **Batch Size**
   - 현재 설정: `max_batch_size: 1`
   - 필요시 증가 가능

4. **Dynamic Batching**
   ```protobuf
   # config.pbtxt에 추가
   dynamic_batching {
     max_queue_delay_microseconds: 100
   }
   ```

## 문제 해결

### 1. "Failed to decode JPEG image" 오류
- JPEG 형식이 올바른지 확인
- 파일이 손상되지 않았는지 확인

### 2. 메모리 부족
- preprocessor 인스턴스 수 감소
- JPEG 품질 낮추기

### 3. 느린 성능
- preprocessor CPU 인스턴스 증가
- JPEG 품질 낮추기 (파일 크기 감소)
- Shared memory 사용

## 벤치마크 예제

```bash
# 100개의 테스트 이미지 생성
python3 create_jpeg_input.py --num-images 100 --quality 90

# 다양한 동시성 수준 테스트
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --input-data=/workspace/kepco/triton/perf_data/input_data.json \
  --concurrency-range 1:16:1 \
  --measurement-interval=10000 \
  --latency-report-file latency_report.json

# Request rate 스윕 테스트
perf_analyzer -m yolov11_ensemble \
  -u localhost:8000 \
  --input-data=/workspace/kepco/triton/perf_data/input_data.json \
  --request-rate-range=50:500:50 \
  --max-threads 64 \
  --measurement-interval=10000
```

## 모니터링

Triton 서버 로그에서 성능 정보 확인:
```bash
# Preprocessor 로그
grep "Preprocessor" /path/to/triton/logs

# YOLOv11 성능 통계 (5초마다)
grep "Performance Stats" /path/to/triton/logs
```

## 주의사항

1. JPEG 디코딩은 CPU에서 수행됩니다 (preprocessor)
2. 실제 추론은 RBLN 디바이스에서 수행됩니다 (yolov11)
3. 대용량 트래픽 시 preprocessor가 병목이 될 수 있습니다
4. 필요시 preprocessor 인스턴스를 더 많이 할당하세요

