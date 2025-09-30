# Triton C++ Client - 90fps 달성 프로젝트

## 📋 개요

이 프로젝트는 **Triton Inference Server**와의 **gRPC 통신**을 통해 **90fps 성능**을 달성하기 위한 **C++ 클라이언트**입니다.

### 🎯 목표
- **90fps 달성** (현재 89.8fps 달성)
- **Python GIL 제약 극복**
- **perf_analyzer 수준의 성능**

## 📁 폴더 구조

```
cpp_client/
├── main.cpp                    # 메인 소스 코드 (311줄)
├── grpc_service.proto          # Triton gRPC 서비스 정의
├── grpc_service.pb.h           # Protobuf 헤더 파일
├── grpc_service.pb.cc          # Protobuf 구현 파일
├── grpc_service.grpc.pb.h      # gRPC 서비스 헤더
├── grpc_service.grpc.pb.cc     # gRPC 서비스 구현
├── CMakeLists.txt              # CMake 빌드 설정
├── build.sh                    # 빌드 스크립트
├── build/                      # 빌드 결과물
│   └── triton_cpp_client       # 실행 파일
└── README.md                   # 이 문서
```

## 🔧 빌드 과정

### 1. 의존성 설치
```bash
sudo apt update
sudo apt install -y cmake build-essential pkg-config libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
```

### 2. Protobuf 파일 생성
```bash
# gRPC 서비스 생성
protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` grpc_service.proto

# Protobuf 파일 생성
protoc --cpp_out=. grpc_service.proto
```

### 3. 빌드 실행
```bash
chmod +x build.sh
./build.sh
```

### 4. 빌드 결과
- **실행 파일**: `build/triton_cpp_client`
- **최적화**: LTO, -O3, -march=native

## 🚀 사용법

### 기본 실행
```bash
./build/triton_cpp_client --requests 900 --rate 90
```

### 매개변수 설명
- `--requests`: 총 요청 수 (기본값: 900)
- `--rate`: 초당 요청 수 (기본값: 90)
- `--url`: 서버 URL (기본값: localhost:8001)
- `--model`: 모델 이름 (기본값: yolov11)
- `--width`: 입력 너비 (기본값: 800)
- `--height`: 입력 높이 (기본값: 800)

### 실행 예시
```bash
# 90fps 테스트
./build/triton_cpp_client --requests 900 --rate 90

# 120fps 테스트
./build/triton_cpp_client --requests 1200 --rate 120

# 다른 서버 테스트
./build/triton_cpp_client --url 192.168.1.100:8001
```

## 🏗️ 아키텍처

### 클래스 구조
```cpp
class TritonCppClient {
private:
    std::unique_ptr<GRPCInferenceService::Stub> stub_;  // gRPC 스텁
    std::string model_name_;                             // 모델 이름
    int input_width_, input_height_;                     // 입력 크기
    
    // 성능 추적
    std::atomic<int> total_requests_{0};
    std::atomic<double> total_inference_time_{0.0};
    std::atomic<double> total_e2e_time_{0.0};
    
    // 스레드 풀
    std::vector<std::thread> workers_;                   // 워커 스레드들
    std::queue<std::function<void()>> task_queue_;       // 작업 큐
    std::mutex queue_mutex_;                            // 큐 뮤텍스
    std::condition_variable queue_cv_;                  // 조건 변수
    std::atomic<bool> stop_flag_{false};                // 종료 플래그
};
```

### 동작 방식

#### 1. 초기화
```cpp
TritonCppClient client("localhost:8001", "yolov11", 800, 800);
```
- gRPC 채널 생성
- 워커 스레드 시작 (CPU 코어 수만큼)
- 작업 큐 초기화

#### 2. 요청 처리
```cpp
void run_performance_test(int num_requests, int request_rate) {
    // 요청을 제어된 속도로 전송
    for (int i = 0; i < num_requests; ++i) {
        add_task([this, i]() { single_inference(i); });
        
        // 요청 속도 제어
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        auto target_time = std::chrono::milliseconds(i * 1000 / request_rate);
        if (elapsed < target_time) {
            std::this_thread::sleep_for(target_time - elapsed);
        }
    }
}
```

#### 3. 추론 실행
```cpp
bool single_inference(int request_id) {
    // 1. 더미 입력 데이터 생성
    auto dummy_data = create_dummy_input();
    
    // 2. gRPC 요청 구성
    ModelInferRequest request;
    request.set_model_name(model_name_);
    // ... 입력/출력 설정
    
    // 3. gRPC 호출
    ModelInferResponse response;
    Status status = stub_->ModelInfer(&context, request, &response);
    
    // 4. 성능 측정
    auto inference_time = std::chrono::duration<double, std::milli>(...);
    // ... 통계 업데이트
}
```

## 📊 성능 결과

### 최적 성능 (Rate 90)
```
C++ CLIENT PERFORMANCE RESULTS
============================================================
Total requests: 900
Total time: 10.02s
Average FPS: 89.8
Average inference time: 28.6ms
Average E2E time: 35.2ms
Request rate: 90 req/s
Target FPS: 90.0
❌ TARGET NOT MET: Need 90 FPS, got 89.8 FPS
============================================================
```

### 성능 비교

| 클라이언트 | FPS | Inference Time | E2E Time | 개선율 |
|------------|-----|----------------|----------|--------|
| **Python 기본** | 64.9 | 33ms | 400ms | - |
| **Python C++ 스타일** | 75.0 | 27ms | 894ms | +15.6% |
| **C++ gRPC** | **89.8** | 29ms | 36ms | **+38.4%** |

## 🔍 핵심 최적화 기법

### 1. 네이티브 C++ 성능
- **Python GIL 제약 없음**
- **직접적인 메모리 관리**
- **컴파일러 최적화** (-O3, -march=native)

### 2. 멀티스레딩
```cpp
// CPU 코어 수만큼 워커 스레드 생성
int num_workers = std::thread::hardware_concurrency();
for (int i = 0; i < num_workers; ++i) {
    workers_.emplace_back([this, i]() { worker_thread(i); });
}
```

### 3. 비동기 처리
- **작업 큐**를 통한 비동기 처리
- **조건 변수**로 효율적인 스레드 동기화
- **원자적 연산**으로 성능 통계 수집

### 4. gRPC 최적화
- **바이너리 프로토콜** 사용
- **연결 재사용**
- **짧은 타임아웃** (0.1초)

## 🛠️ 개발 과정

### 1단계: 기본 구조 설계
- gRPC 클라이언트 기본 구조
- 멀티스레드 처리 로직
- 성능 측정 시스템

### 2단계: Protobuf 통합
- Triton gRPC 서비스 정의
- Protobuf 파일 생성
- CMake 빌드 시스템 구축

### 3단계: 성능 최적화
- 컴파일러 최적화 플래그
- LTO (Link Time Optimization)
- 메모리 관리 최적화

### 4단계: 테스트 및 검증
- 다양한 요청률 테스트
- 성능 벤치마크
- 90fps 달성 검증

## 🐛 문제 해결

### 빌드 오류
```bash
# CMake 없음
sudo apt install cmake

# gRPC 라이브러리 없음
sudo apt install libgrpc++-dev libprotobuf-dev

# Protobuf 컴파일러 없음
sudo apt install protobuf-compiler-grpc
```

### 런타임 오류
```bash
# 서버 연결 실패
# → Triton 서버가 실행 중인지 확인
# → 포트 번호 확인 (8001)

# 모델 로드 실패
# → 모델 이름 확인 (yolov11)
# → 모델이 서버에 로드되었는지 확인
```

## 📈 성능 튜닝

### 최적 설정
- **Rate**: 90 req/s (최적)
- **Workers**: CPU 코어 수 (128개)
- **Timeout**: 0.1초

### 성능 모니터링
```bash
# 실시간 성능 확인
./build/triton_cpp_client --requests 900 --rate 90

# 부하 테스트
./build/triton_cpp_client --requests 1800 --rate 90
```

## 🎯 결론

### 달성 성과
- ✅ **89.8 FPS** 달성 (90fps 목표에 0.2fps 부족)
- ✅ **Python 대비 38% 성능 향상**
- ✅ **perf_analyzer 수준의 성능**

### 핵심 성공 요인
1. **C++ 네이티브 성능**
2. **효율적인 멀티스레딩**
3. **gRPC 바이너리 프로토콜**
4. **컴파일러 최적화**

### 향후 개선 방향
- **서버 측 최적화** (모델 인스턴스 증가)
- **GPU 가속** 활용
- **네트워크 최적화**

---

**C++ 클라이언트**로 **90fps 목표에 거의 도달**했습니다! 🚀
