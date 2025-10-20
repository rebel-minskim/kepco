# NPU vs GPU 성능 비교 대시보드

AI 영상 처리 워크로드에 대한 ATOM™-Max NPU와 NVIDIA L40S GPU 성능 지표를 비교하는 실시간 웹 대시보드입니다.

![대시보드 미리보기](image.png)

![상태](https://img.shields.io/badge/Status-Live-green)

## 주요 기능

- **듀얼 비디오 비교**: NPU와 GPU 처리 결과를 나란히 표시하는 비디오 재생
- **실시간 전력 모니터링**: 현재 전력 소비를 보여주는 애니메이션 게이지 차트
- **성능 효율성 추적**: FPS per Watt를 비교하는 실시간 스크롤 그래프
- **전력 사용량 분석**: 와트(W) 단위의 실시간 소비 전력 시각화
- **자동 계산 지표**: 데이터로부터 효율성 배수를 자동으로 계산

## 대시보드 구성 요소

### 1. 비디오 패널 (상단)
- **왼쪽**: ATOM™-Max NPU 처리 출력 (36 Imgs/s)
- **오른쪽**: L40S GPU 처리 출력 (24 Imgs/s)
- 두 비디오가 동기화되어 재생됩니다

### 2. 전력 소비 (하단 왼쪽)
- 실시간 전력 사용량을 보여주는 게이지 차트
- 데이터 스트림에 따라 동적으로 업데이트
- 현재 전력 및 최대 용량 표시

### 3. 성능 효율성 (하단 중앙)
- 시간 경과에 따른 FPS/Watt를 보여주는 선 그래프
- 오른쪽에서 왼쪽으로 스크롤
- 효율성 이점 배수 표시 (예: "6.3x")

### 4. 에너지 소비 (하단 오른쪽)
- 시간 경과에 따른 전력 사용량을 보여주는 선 그래프
- 실시간 와트 측정
- 에너지 효율성 비교

## 빠른 시작

### 사전 요구사항
- 최신 웹 브라우저 (Chrome, Firefox, Safari, Edge)
- 비디오 파일: `output_npu.mp4` 및 `output_gpu.mp4`
- 데이터 파일: 전력 및 FPS 지표가 포함된 JSON 파일

### 두 가지 사용 방법:

#### 옵션 1: 독립 실행형 (오프라인 사용 권장)

**서버 불필요! 완전히 오프라인으로 작동합니다.**

1. **독립 실행형 버전 빌드:**
   ```bash
   cd cctv_solution/frontend
   python3 build_standalone.py
   ```

2. **파일 열기:**
   - `index_standalone.html`을 **더블클릭** 하거나
   - 오른쪽 클릭 → 연결 프로그램 → 브라우저 선택

3. **완료!** 인터넷 없이 오프라인으로 작동합니다.

**사용 시점:**
- 오프라인 프레젠테이션 (인터넷 없음)
- 매번 서버를 시작하고 싶지 않을 때
- 간단한 더블클릭으로 열기
- 단일 HTML 파일 공유 (+ 비디오)

#### 옵션 2: 웹 서버 사용

**개발 및 빈번한 데이터 업데이트에 더 적합합니다.**

1. **프론트엔드 디렉토리로 이동:**
   ```bash
   cd cctv_solution/frontend
   ```

2. **로컬 웹 서버 시작:**
   ```bash
   python3 -m http.server 8080
   ```

3. **브라우저 열기:**
   ```
   http://localhost:8080/index.html
   ```

**사용 시점:**
- 개발 및 테스트
- 데이터를 자주 업데이트할 때
- 변경 후 매번 재빌드하고 싶지 않을 때

### 대안: Node.js 사용

Node.js를 선호하는 경우:
```bash
npx http-server -p 8080
```

### 필수 파일

```
frontend/
├── index.html              # 원본 (서버 필요)
├── index_standalone.html   # 독립 실행형 (서버 불필요)
├── script.js
├── style.css
├── build_standalone.py     # 빌드 스크립트
├── output_npu.mp4
├── output_gpu.mp4
├── npu_power.json
├── npu_fps.json
├── gpu_power.json
└── gpu_fps.json
```

## 사용 방법

1. **페이지 로드**: 브라우저에서 `http://localhost:8080/index.html` 열기

2. **애니메이션 시작**: 비디오에서 재생을 클릭하여 시작:
   - 비디오가 동기화되어 재생됩니다
   - 차트가 애니메이션을 시작합니다
   - 전력 게이지가 실시간으로 업데이트됩니다

3. **지표 관찰**:
   - 게이지 차트에서 전력 소비 변화 확인
   - 효율성 그래프가 오른쪽에서 왼쪽으로 스크롤되는 것을 확인
   - NPU(녹색)와 GPU(보라색) 선 비교

4. **일시 정지/재개**: 비디오에서 일시 정지를 클릭하여 모든 애니메이션 정지

5. **루프**: 비디오와 데이터가 자동으로 반복되어 지속적인 모니터링 가능

## 데이터 업데이트 (독립 실행형 버전)

독립 실행형 버전을 사용 중이고 데이터를 업데이트하려는 경우:

```bash
# 1. JSON 데이터 파일 업데이트
# npu_power.json, gpu_power.json, npu_fps.json, gpu_fps.json 편집

# 2. 독립 실행형 파일 재빌드
python3 build_standalone.py

# 3. 새로운 독립 실행형 파일 열기
# index_standalone.html 더블클릭
```

**완료!** 모든 새 데이터가 HTML 파일에 포함됩니다.

## 데이터 커스터마이징

### 1단계: 비디오 파일 준비

비디오를 웹 호환 형식으로 변환:
```bash
# NPU 비디오
ffmpeg -i your_npu_video.mp4 -c:v libx264 -preset fast -crf 23 \
  -movflags +faststart output_npu.mp4

# GPU 비디오
ffmpeg -i your_gpu_video.mp4 -c:v libx264 -preset fast -crf 23 \
  -movflags +faststart output_gpu.mp4
```

### 2단계: 전력 데이터 JSON 생성

형식: `npu_power.json` 및 `gpu_power.json`

```json
{
  "metadata": {
    "total_samples": 41,
    "duration_seconds": 23.9,
    "device": "장치 이름"
  },
  "samples": [
    {
      "timestamp": 1760605347.19,
      "power": 50.5,
      "relative_time": 0.0
    },
    {
      "timestamp": 1760605347.73,
      "power": 51.2,
      "relative_time": 0.54
    }
    // ... 더 많은 샘플
  ]
}
```

**주요 필드:**
- `samples[].power`: 와트 단위의 전력
- `samples[].timestamp`: Unix 타임스탬프
- `samples[].relative_time`: 시작으로부터의 초

### 3단계: FPS 데이터 JSON 생성

형식: `npu_fps.json` 및 `gpu_fps.json`

```json
{
  "metadata": {
    "total_samples": 180
  },
  "summary": {
    "total_frames": 1800,
    "total_time_seconds": 23.36,
    "average_fps": 90.0,
    "video_info": {
      "width": 1920,
      "height": 1080
    }
  },
  "statistics": {
    "inference_time": {
      "mean_ms": 28.0,
      "min_ms": 18.0,
      "max_ms": 45.0
    }
  },
  "frames": []
}
```

**주요 필드:**
- `summary.average_fps`: 평균 초당 프레임 수
- `summary.total_time_seconds`: 처리 시간

### 4단계: 장치 이름 업데이트 (선택사항)

`index.html`을 편집하여 장치 이름 변경:
```html
<!-- Line 19 -->
<div class="device-name">NPU 이름</div>

<!-- Line 43 -->
<div class="device-name">GPU 이름</div>
```

### 5단계: 새로고침 및 확인

페이지를 새로고침하여 데이터를 확인하세요!

## 계산된 지표

대시보드는 자동으로 다음을 계산합니다:

### 성능 효율성
```
효율성 = FPS / 평균 전력 (와트)
예: 90 FPS / 50W = 1.8 FPS/Watt
```

### 프레임당 에너지
```
에너지 = 평균 전력 / FPS
예: 50W / 90 FPS = 0.556 줄/프레임
```

### 효율성 배수
```
배수 = NPU 효율성 / GPU 효율성
NPU가 몇 배 더 효율적인지 표시
```

## 문제 해결

### 문제: CORS 오류 (크로스 오리진 요청 차단됨)

**문제**: `index.html`을 직접 열면 CORS 오류가 표시됨

**해결책**: 항상 웹 서버 사용
```bash
# Python 사용
python3 -m http.server 8080

# 또는 Node.js 사용
npx http-server -p 8080
```

### 문제: 비디오가 재생되지 않음

**문제**: 비디오에 검은 화면이 표시되거나 로드되지 않음

**해결책**:
1. 비디오 코덱 확인: H.264여야 함
   ```bash
   ffmpeg -i video.mp4 -c:v libx264 output.mp4
   ```

2. HTML의 파일 경로가 비디오 파일 이름과 일치하는지 확인

3. 브라우저 콘솔(F12)에서 오류 메시지 확인

### 문제: 차트가 표시되지 않음

**문제**: 차트가 있어야 할 곳에 빈 영역이 표시됨

**해결책**:
1. 브라우저 콘솔(F12)을 열고 오류 확인
2. JSON 파일이 유효한지 확인 (jsonlint.com 사용)
3. Chart.js가 로드되었는지 확인: 콘솔에서 "Script loaded" 찾기
4. 데이터 파일이 존재하고 올바른 이름인지 확인

### 문제: 차트가 애니메이션되지 않음

**문제**: 차트는 보이지만 움직이지 않음

**해결책**:
1. 비디오에서 재생을 클릭
2. 콘솔에서 "Animation loop started" 메시지 확인
3. 데이터 파일에 여러 샘플이 있는지 확인 (최소 10개 이상)
4. JSON 파일에 `samples` 배열이 있는지 확인

### 문제: 잘못된 지표 표시

**문제**: 효율성 배수가 잘못된 값을 표시함

**해결책**:
1. FPS JSON의 `summary` 섹션에 `average_fps`가 있는지 확인
2. 전력 값이 와트 단위인지 확인 (밀리와트가 아님)
3. 샘플 수가 0보다 큰지 확인
4. 재계산: 콘솔을 열고 로그된 지표 확인

## 데이터 수집 팁

### 전력 데이터 수집

**NVIDIA GPU의 경우:**
```bash
# 지속적인 모니터링
nvidia-smi --query-gpu=timestamp,power.draw \
  --format=csv,noheader,nounits --loop-ms=500 > gpu_power.txt
```

**NPU의 경우:** 장치의 모니터링 도구를 사용하여 유사한 데이터 수집

### FPS 데이터 수집

비디오 처리 중 다음 지표 추적:
- 처리된 총 프레임 수
- 총 처리 시간
- 평균 FPS: `총_프레임 / 총_시간`
- 프레임당 추론 시간

### 비디오와 데이터 동기화

최상의 결과를 위해:
- 비디오 처리 시작 시 전력 모니터링 시작
- 처리 종료 시 전력 모니터링 중지
- 데모용으로 처리된 것과 동일한 비디오 파일 사용
- 데이터 시간과 비디오 시간 일치

## 커스터마이징

### 색상 변경

`style.css` 편집:
```css
/* NPU 색상 (녹색) */
.legend-item.atom .legend-dot { background: #76ff03; }

/* GPU 색상 (보라색) */
.legend-item.nvidia .legend-dot { background: #b794f6; }
```

### 애니메이션 속도 조정

`script.js` 편집:
```javascript
// Line 350: 업데이트 간격 계산 변경
let updateInterval = 500; // 밀리초
```

### 데이터 윈도우 크기 변경

`script.js` 편집:
```javascript
// Line 12: 표시되는 데이터 포인트 수
maxDataPoints: 60  // 마지막 60개 샘플 표시
```

## 파일 구조

```
frontend/
├── index.html          # 메인 HTML 페이지
├── script.js           # JavaScript 로직 및 애니메이션
├── style.css           # 스타일 및 레이아웃
├── README.md           # 이 파일
├── DATA_FORMAT.md      # 상세 데이터 형식 사양
│
├── Videos (필수):
│   ├── output_npu.mp4  # NPU 처리 비디오
│   └── output_gpu.mp4  # GPU 처리 비디오
│
└── Data Files (필수):
    ├── npu_power.json  # NPU 전력 샘플
    ├── npu_fps.json    # NPU FPS 데이터
    ├── gpu_power.json  # GPU 전력 샘플
    └── gpu_fps.json    # GPU FPS 데이터
```

## 브라우저 콘솔 명령

콘솔(F12)을 열고 다음을 시도하세요:

```javascript
// 로드된 데이터 확인
console.log(gpuPowerData);
console.log(gpuFpsData);

// 지표 확인
console.log(calculateMetrics());

// 애니메이션 수동 중지
stopAnimationLoop();

// 애니메이션 수동 시작
startAnimationLoop();
```

## 요구사항

- **브라우저**: ES6를 지원하는 최신 브라우저
- **웹 서버**: Python 3.x, Node.js 또는 HTTP 서버
- **비디오 코덱**: H.264 (MP4)
- **데이터 형식**: 지정된 구조의 JSON 파일

## 제한사항

- 로컬 웹 서버 필요 (file:// 직접 열기 불가)
- 비디오는 H.264로 인코딩되어야 함
- 권장 최대 데이터 포인트: ~1000 샘플
- 데스크톱/노트북에서 가장 잘 보임 (반응형 디자인 제한적)

## 최상의 결과를 위한 팁

1. **비디오 품질**: 고품질 H.264 인코딩 비디오 사용
2. **데이터 동기화**: 데이터 시간이 비디오 시간과 일치하는지 확인
3. **샘플링 속도**: 초당 2-5개 샘플이 부드러운 애니메이션 제공
4. **파일 크기**: 더 나은 로딩을 위해 비디오를 100MB 이하로 유지
5. **테스트**: 배포 전 항상 로컬에서 테스트

## 지원

**디버그 체크리스트:**
- [ ] 웹 서버가 실행 중
- [ ] 모든 파일이 frontend 디렉토리에 존재
- [ ] JSON 파일이 유효함
- [ ] 비디오가 H.264로 인코딩됨
- [ ] 브라우저 콘솔에 오류가 없음
- [ ] 포트가 방화벽에 의해 차단되지 않음

**여전히 문제가 있나요?**
1. 브라우저 콘솔(F12)에서 오류 메시지 확인
2. 파일 이름이 정확히 일치하는지 확인 (대소문자 구분)
3. 먼저 샘플 데이터로 테스트
4. 모든 JSON 파일이 올바른 구조를 가지고 있는지 확인

## 라이선스

Copyright © 2025 Rebellions Inc.

---

**빠른 명령 참조:**

```bash
# 서버 시작
python3 -m http.server 8080

# 비디오 변환
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# JSON 검증
python3 -m json.tool data.json

# 브라우저에서 열기
open http://localhost:8080/index.html
```

**AI 성능 분석을 위해 제작됨**

