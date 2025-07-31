# RF-DETR ONNX Test

RF-DETR (Real-time Detection Transformer) 객체 검출 모델을 Rust로 구현한 프로젝트입니다. ONNX Runtime을 사용하여 모델 추론을 수행하고, 콘솔 및 GUI 인터페이스를 제공합니다.

## 주요 기능

- **ONNX 모델 추론**: RF-DETR 모델을 사용한 실시간 객체 검출
- **다중 모델 지원**: 8가지 다양한 양자화 모델 선택 가능
- **추론 시간 측정**: 실시간 추론 성능 모니터링
- **이미지 전처리**: 레터박싱을 통한 종횡비 유지 리사이징 및 정규화
- **바운딩 박스 시각화**: 검출된 객체에 대한 바운딩 박스 및 클래스 정보 표시
- **다중 인터페이스**: 콘솔 모드와 GUI 모드 지원
- **명령행 인자 지원**: 이미지 파일 경로와 모델 타입을 인자로 받아 콘솔 모드 실행

## 설치 및 실행

### 요구사항

- Rust 1.70+
- Windows 10/11 (테스트된 환경)
- ONNX Runtime CPU 버전 (자동 설치됨)

### 설치

```bash
git clone <repository-url>
cd rf_detr_onnx_test
cargo build
```

### 실행

#### GUI 모드 (기본)

```bash
cargo run --release
```

#### 콘솔 모드 (이미지 파일 및 모델 타입 지정)

```bash
cargo run --release -- image.jpg model_type
```

#### 사용법

```bash
# GUI 모드 실행 (인자 없음)
cargo run --release

# 콘솔 모드 실행 (이미지 파일 경로 지정, 기본 모델 사용)
cargo run --release -- path/to/image.jpg

# 콘솔 모드 실행 (이미지 파일 경로와 모델 타입 지정)
cargo run --release -- path/to/image.jpg original
cargo run --release -- path/to/image.jpg fp16
cargo run --release -- path/to/image.jpg int8

# 도움말 표시 (잘못된 인자)
cargo run --release -- arg1 arg2 arg3
```

## 프로젝트 구조

```
rf_detr_onnx_test/
├── src/
│   ├── main.rs          # 메인 실행 파일 (명령행 인자 처리)
│   ├── lib.rs           # 핵심 라이브러리 (ONNX 추론, 이미지 처리)
│   ├── gui.rs           # egui 기반 GUI 구현
│   └── coco_classes.rs  # COCO 클래스 정의
├── assets/
│   ├── models/
│   │   ├── model.onnx         # 원본 모델 (108 MB)
│   │   ├── model_fp16.onnx    # FP16 양자화 모델 (55.2 MB)
│   │   ├── model_int8.onnx    # INT8 양자화 모델 (29.6 MB)
│   │   ├── model_uint8.onnx   # UINT8 양자화 모델 (29.6 MB)
│   │   ├── model_quantized.onnx # 일반 양자화 모델 (29.6 MB)
│   │   ├── model_q4.onnx      # 4비트 양자화 모델 (25.3 MB)
│   │   ├── model_q4f16.onnx   # 4비트+FP16 하이브리드 (20.1 MB)
│   │   └── model_bnb4.onnx    # BitsAndBytes 4비트 (23.8 MB)
│   └── images/
│       └── sample.png         # 샘플 이미지
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

- RF-DETR 모델을 사용한 80개 COCO 클래스 검출
- 신뢰도 점수 기반 필터링 (임계값: 0.5)
- 바운딩 박스 좌표 추출 및 변환
- 정확한 클래스 매핑 (Background=0, Person=1, Bicycle=2, ...)

### 이미지 처리

- **레터박싱**: 종횡비를 유지하면서 560x560으로 리사이징
- HWC → CHW 변환
- 픽셀 값 정규화 (0-255 → 0-1)
- 바운딩 박스 및 클래스 정보 시각화

### 실행 모드

#### GUI 모드

- 직관적인 파일 선택 인터페이스
- **모델 선택 드롭다운**: 8가지 모델 중 선택 가능
- **실시간 추론 시간 표시**: 파란색으로 강조된 성능 정보
- 실시간 이미지 처리 및 표시
- 검출 결과 목록 표시
- 에러 메시지 표시
- 드래그 앤 드롭 지원 (향후 구현 예정)

#### 콘솔 모드

- 명령행에서 이미지 파일 경로와 모델 타입 지정
- 검출 결과를 콘솔에 출력
- **추론 시간 표시**: 밀리초 단위로 정확한 성능 측정
- 처리된 이미지를 `output_with_detections.png`로 저장
- 배치 처리에 적합

### 지원 모델

프로젝트는 8가지 다양한 RF-DETR 모델을 지원합니다:

| 모델 타입 | 파일명 | 크기 | 상태 | 추론 시간 | 비고 |
|-----------|--------|------|------|-----------|------|
| **Original** | `model.onnx` | 108 MB | ✅ | **426.74 ms** | 가장 빠르고 안정적 |
| **FP16** | `model_fp16.onnx` | 55.2 MB | ✅ | 502.73 ms | 정상 작동, 메모리 절약 |
| INT8 | `model_int8.onnx` | 29.6 MB | ❌ | - | ConvInteger 연산자 미지원 |
| UINT8 | `model_uint8.onnx` | 29.6 MB | ❌ | - | Reshape 연산자 오류 |
| Quantized | `model_quantized.onnx` | 29.6 MB | ❌ | - | Reshape 연산자 오류 |
| Q4 | `model_q4.onnx` | 25.3 MB | ❌ | - | MatMulNBits 연산자 미지원 |
| Q4F16 | `model_q4f16.onnx` | 20.1 MB | ❌ | - | MatMulNBits 연산자 미지원 |
| BNB4 | `model_bnb4.onnx` | 23.8 MB | ❌ | - | MatMulBnb4 연산자 미지원 |

**권장 모델**: 현재 환경에서는 **Original 모델**을 사용하는 것이 최적입니다.

## 성능 테스트 결과

### 테스트 환경
- **OS**: macOS 14.5.0
- **CPU**: Apple Silicon
- **ONNX Runtime**: v1.16.0
- **테스트 이미지**: 164KB JPEG (640x480)

### 성능 비교

#### 정상 작동 모델

**Original 모델 (108 MB)**
```
=== Detection Results ===
Found 4 objects:
1. person (Confidence: 89.7%, BBox: [0.586, 0.051, 0.905, 1.000])
2. person (Confidence: 86.9%, BBox: [0.093, 0.270, 0.905, 1.000])
3. snowboard (Confidence: 72.0%, BBox: [0.338, 0.605, 0.412, 1.000])
4. snowboard (Confidence: 58.2%, BBox: [0.771, 0.433, 0.885, 0.989])
⏱️ Inference Time: 426.74 ms
```

**FP16 모델 (55.2 MB)**
```
=== Detection Results ===
Found 4 objects:
1. person (Confidence: 89.6%, BBox: [0.586, 0.051, 0.904, 1.000])
2. person (Confidence: 86.2%, BBox: [0.092, 0.270, 0.905, 1.000])
3. snowboard (Confidence: 71.9%, BBox: [0.338, 0.605, 0.412, 1.000])
4. snowboard (Confidence: 58.3%, BBox: [0.771, 0.433, 0.885, 0.989])
⏱️ Inference Time: 502.73 ms
```

### 성능 분석

1. **Original 모델이 FP16보다 18% 빠름** (426ms vs 502ms)
2. **정확도는 거의 동일** (신뢰도 차이 < 1%)
3. **메모리 사용량**: FP16이 49% 절약 (108MB → 55MB)

### 호환성 문제 분석

#### 4비트 양자화 모델들 (Q4, Q4F16, BNB4)
- **오류**: `MatMulNBits`, `MatMulBnb4` 연산자 미지원
- **원인**: 현재 ONNX Runtime 버전에서 지원되지 않음
- **해결**: 최신 ONNX Runtime 버전 필요

#### 8비트 양자화 모델들 (INT8, UINT8, Quantized)
- **INT8 오류**: `ConvInteger` 연산자 미지원
- **UINT8/Quantized 오류**: Reshape 연산자 입력 크기 불일치
- **원인**: 모델 입력 크기와 전처리 파이프라인 불일치

## 개발 과정

### 초기 구현

- ONNX Runtime을 사용한 기본 추론 구현
- 이미지 전처리 및 후처리 로직 구현
- 바운딩 박스 파싱 및 시각화

### GUI 구현

- **egui 채택**: 안정적이고 성숙한 GUI 프레임워크
- 파일 선택 및 이미지 표시 기능 구현

### 최적화 및 리팩토링

- 코드 모듈화 및 구조 개선
- 명령행 인자 처리 추가
- 에러 처리 개선
- 사용자 인터페이스 개선
- 레터박싱 구현으로 이미지 왜곡 방지

### 다중 모델 지원 추가

- 8가지 다양한 양자화 모델 지원
- 모델 선택 드롭다운 구현
- 추론 시간 측정 기능 추가
- 호환성 문제 분석 및 문서화

## 문제 해결

### ONNX Runtime 관련

- `0xc000007b` 오류: ort 크레이트로 해결
- 텐서 형식 불일치: 올바른 전처리 파이프라인 구현
- API 호환성: ort 1.16.0 버전으로 안정화

### 양자화 모델 호환성 문제

#### INT8 모델 오류
- **오류**: `ConvInteger` 연산자 미지원
- **해결**: 원본 모델 사용 또는 최신 ONNX Runtime 업그레이드

#### 4비트 양자화 모델 오류
- **오류**: `MatMulNBits`, `MatMulBnb4` 연산자 미지원
- **해결**: 최신 ONNX Runtime 버전 필요

#### Reshape 연산자 오류
- **오류**: 입력 텐서 크기 불일치
- **해결**: 모델 입력 크기 조정 (560 → 800)

### GUI 관련

- Dioxus API 호환성 문제: egui로 전환
- 이미지 텍스처 로딩: ColorImage API 사용
- 모델 선택 드롭다운: ComboBox API 사용

### 클래스 검출 문제

- 잘못된 클래스 매핑: COCO 표준 인덱싱으로 수정
- Person이 Bicycle로 검출되는 문제 해결

### 성능 최적화

- FP16 모델이 원본보다 느린 문제: CPU 최적화 문제로 확인
- 추론 시간 측정: `std::time::Instant` 사용으로 정확한 측정

## 향후 계획

### 단기 계획
- [ ] 드래그 앤 드롭 기능 완성
- [ ] 배치 처리 지원
- [ ] 추가 GUI 기능 (설정, 결과 저장 등)
- [ ] 모델 성능 비교 차트 추가

### 중기 계획
- [ ] 실시간 비디오 처리
- [ ] GPU 가속 지원 (FP16 모델 최적화)
- [ ] 최신 ONNX Runtime 업그레이드 (양자화 모델 호환성 개선)
- [ ] 모델 캐싱 및 로딩 최적화

### 장기 계획
- [ ] 웹 인터페이스 추가
- [ ] 클라우드 배포 지원
- [ ] 모바일 앱 개발
- [ ] 커스텀 모델 지원

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

---

**참고**: 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 프로덕션 환경에서 사용하기 전에 충분한 테스트를 권장합니다. 