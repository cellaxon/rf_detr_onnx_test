# RF-DETR ONNX Test

RF-DETR (Real-time Detection Transformer) 객체 검출 모델을 Rust로 구현한 프로젝트입니다. ONNX Runtime을 사용하여 모델 추론을 수행하고, 직관적인 GUI 인터페이스를 제공합니다.

## 주요 기능

- **ONNX 모델 추론**: RF-DETR 원본 모델을 사용한 실시간 객체 검출
- **모델 캐싱**: 빠른 추론을 위한 모델 세션 캐싱
- **추론 시간 측정**: 실시간 추론 성능 모니터링
- **이미지 전처리**: 레터박싱을 통한 종횡비 유지 리사이징 및 정규화
- **바운딩 박스 시각화**: 검출된 객체에 대한 바운딩 박스 및 클래스 정보 표시
- **GUI 인터페이스**: 직관적인 사용자 인터페이스
- **분할 레이아웃**: 좌측 검출 결과, 우측 이미지 표시
- **스크롤 지원**: 양쪽 패널 모두 스크롤 가능
- **드래그 앤 드롭**: 이미지 파일을 직접 드래그하여 로드

## 설치 및 실행

### 요구사항

- Rust 1.70+
- macOS 14.5.0+ (테스트된 환경)
- ONNX Runtime

### 설치

#### 1. ONNX Runtime 설치
```bash
pip install onnxruntime
```

#### 2. 프로젝트 클론
```bash
git clone https://github.com/cellaxon/rf_detr_onnx_test
cd rf_detr_onnx_test
```

#### 3. 모델 다운로드

프로젝트를 실행하기 전에 RF-DETR 모델을 다운로드해야 합니다:

1. [Hugging Face RF-DETR ONNX 모델 페이지](https://huggingface.co/onnx-community/rfdetr_base-ONNX/tree/main/onnx)에 접속
2. `model.onnx` 파일을 다운로드 (108 MB)
3. 다운로드한 파일을 `assets/models/` 폴더에 `model.onnx` 이름으로 저장

```bash
# assets/models 폴더 생성 (없는 경우)
mkdir -p assets/models

# 모델 파일을 assets/models/model.onnx로 이동
mv ~/Downloads/model.onnx assets/models/
```

**동작 가능한 모델** (macOS M4에서 테스트됨):
- ✅ **원본 모델** (`model.onnx`, 108 MB): 가장 빠른 추론 속도
- ✅ **FP16 모델** (`model_fp16.onnx`, 55.2 MB): 메모리 절약하지만 느림
- ❌ **INT8/UINT8 모델**: `ConvInteger` 연산자 미지원으로 동작 불가
- ❌ **4비트 양자화 모델**: `MatMulNBits` 연산자 미지원으로 동작 불가

**성능 비교** (macOS M4 기준): 원본 모델이 FP16 모델보다 약 18% 빠른 추론 속도를 보입니다 (426ms vs 502ms).

#### 4. 프로젝트 빌드
```bash
cargo build
```

### 실행

```bash
cargo run --release
```

## 프로젝트 구조

```
rf_detr_onnx_test/
├── src/
│   ├── main.rs          # 메인 실행 파일
│   ├── lib.rs           # 핵심 라이브러리 (ONNX 추론, 이미지 처리)
│   └── gui.rs           # egui 기반 GUI 구현
├── assets/
│   └── models/
│       └── model.onnx   # RF-DETR 원본 모델 (108 MB)
├── Cargo.toml
└── README.md
```

## 사용된 기술

### 핵심 라이브러리

- **ort**: ONNX Runtime Rust 바인딩 (v1.16.0)
- **image**: 이미지 처리 및 변환
- **ndarray**: 다차원 배열 연산
- **imageproc**: 이미지 처리 및 그리기

### GUI 라이브러리

- **egui**: 즉시 모드 GUI 프레임워크
- **eframe**: egui 애플리케이션 프레임워크
- **rfd**: 파일 다이얼로그

### 기타

- **anyhow**: 에러 처리

## 기능 상세

### 객체 검출

- RF-DETR 모델을 사용한 90개 클래스 검출
- 신뢰도 점수 기반 필터링 (임계값: 0.5)
- 바운딩 박스 좌표 추출 및 변환
- 정확한 RF-DETR 클래스 매핑 (Person=1, Cat=17, Dog=18, Horse=19, ...)

### 이미지 처리

- **레터박싱**: 종횡비를 유지하면서 560x560 리사이징
- HWC → CHW 변환
- 픽셀 값 정규화 (0-255 → 0-1)
- 바운딩 박스 및 클래스 정보 시각화

### GUI 인터페이스

- **분할 레이아웃**: 좌측 검출 결과, 우측 이미지 표시
- **스크롤 지원**: 양쪽 패널 모두 스크롤 가능
- **드래그 앤 드롭**: 이미지 파일을 직접 드래그하여 로드
- **모델 정보 표시**: 현재 사용 중인 모델 정보
- **실시간 추론 시간 표시**: 파란색으로 강조된 성능 정보
- **에러 처리**: 안전한 오류 처리 및 사용자 피드백

## 프로그램 실행 흐름

### 애플리케이션 시작 및 이미지 처리 흐름

```mermaid
flowchart TD
    A[프로그램 시작] --> B[GUI 초기화]
    B --> C[메인 윈도우 생성]
    C --> D[좌측 패널: 검출 결과]
    C --> E[우측 패널: 이미지 표시]
    
    F[이미지 선택 방법] --> G{선택 방식}
    G -->|파일 선택 버튼| H[파일 다이얼로그 열기]
    G -->|드래그 앤 드롭| I[이미지 파일 드롭]
    
    H --> J[이미지 파일 선택]
    I --> J
    J --> K[이미지 파일 읽기]
    K --> L{파일 읽기 성공?}
    L -->|실패| M[에러 메시지 표시]
    L -->|성공| N[ModelCache 초기화]
    
    N --> O{모델 캐시 생성 성공?}
    O -->|실패| P[캐시 초기화 에러 표시]
    O -->|성공| Q[이미지 전처리 시작]
    
    Q --> R[레터박싱 리사이징]
    R --> S[HWC → CHW 변환]
    S --> T[픽셀 정규화 0-1]
    T --> U[ONNX 모델 추론]
    
    U --> V[추론 시간 측정]
    V --> W[모델 출력 파싱]
    W --> X[바운딩 박스 좌표 추출]
    X --> Y[클래스 확률 계산]
    Y --> Z[신뢰도 필터링 > 0.5]
    
    Z --> AA[검출 결과 생성]
    AA --> BB[바운딩 박스 그리기]
    BB --> CC[결과 이미지 생성]
    CC --> DD[GUI 텍스처 로딩]
    
    DD --> EE[좌측 패널: 검출 결과 표시]
    DD --> FF[우측 패널: 처리된 이미지 표시]
    DD --> GG[추론 시간 표시]
    
    M --> HH[에러 상태 유지]
    P --> HH
    HH --> II[사용자 재시도 대기]
    II --> F
    
    EE --> JJ[사용자 다음 작업 대기]
    FF --> JJ
    GG --> JJ
    JJ --> F
```

### 상세 처리 단계

#### 1. 이미지 전처리 단계
```mermaid
flowchart LR
    A[원본 이미지] --> B[종횡비 계산]
    B --> C[레터박싱 리사이징]
    C --> D[560x560 정사각형 캔버스]
    D --> E[회색 패딩 추가]
    E --> F[HWC → CHW 변환]
    F --> G[픽셀 정규화 0-1]
    G --> H[ONNX 입력 텐서]
```

#### 2. 모델 추론 및 후처리 단계
```mermaid
flowchart LR
    A[ONNX 모델 입력] --> B[RF-DETR 추론]
    B --> C[바운딩 박스 텐서]
    B --> D[클래스 로짓 텐서]
    C --> E[좌표 파싱]
    D --> F[시그모이드 적용]
    F --> G[신뢰도 계산]
    E --> H[레터박싱 좌표 변환]
    G --> I[임계값 필터링 > 0.5]
    H --> J[검출 결과 생성]
    I --> J
    J --> K[바운딩 박스 그리기]
    K --> L[최종 결과 이미지]
```

## 성능

### 테스트 환경
- **OS**: macOS 14.5.0
- **CPU**: Apple Silicon
- **ONNX Runtime**: v1.16.0
- **테스트 이미지**: 164KB JPEG (640x480)

### 성능 결과

**RF-DETR Original 모델 (108 MB)**
```
Found 4 objects:
1. person (Confidence: 89.7%, BBox: [0.586, 0.051, 0.905, 1.000])
2. person (Confidence: 86.9%, BBox: [0.093, 0.270, 0.905, 1.000])
3. snowboard (Confidence: 72.0%, BBox: [0.338, 0.605, 0.412, 1.000])
4. snowboard (Confidence: 58.2%, BBox: [0.771, 0.433, 0.885, 0.989])
⏱️ Inference Time: 426.74 ms
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

---

**참고**: 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 프로덕션 환경에서 사용하기 전에 충분한 테스트를 권장합니다. 