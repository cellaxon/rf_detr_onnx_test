# RF-DETR Object Detection with Rust

이 프로젝트는 Rust를 사용하여 RF-DETR (Real-time Detection Transformer) 모델을 ONNX Runtime으로 실행하는 객체 검출 애플리케이션입니다.

## 🚀 기능

- **RF-DETR 모델**: 실시간 객체 검출을 위한 최신 Transformer 기반 모델
- **ONNX Runtime**: 크로스 플랫폼 추론 엔진
- **임베디드 리소스**: 모델과 샘플 이미지가 실행 파일에 포함됨
- **바운딩 박스 시각화**: 검출된 객체에 빨간색 박스 표시
- **COCO 클래스 지원**: 80개 COCO 객체 클래스 인식
- **신뢰도 점수**: 각 검출에 대한 신뢰도 표시

## 📋 요구사항

- Rust 1.88.0 이상
- Windows 10/11 (테스트됨)
- ONNX Runtime CPU 버전

## 🛠️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd rf_detr_onnx_test
```

### 2. 의존성 설치
```bash
cargo build --release
```

### 3. 실행
```bash
cargo run --release
```

## 📁 프로젝트 구조

```
rf_detr_onnx_test/
├── assets/
│   ├── models/
│   │   └── rf-detr-base.onnx    # RF-DETR 모델 파일
│   └── images/
│       └── sample.png           # 샘플 이미지
├── src/
│   ├── main.rs                  # 메인 실행 파일
│   ├── lib.rs                   # 라이브러리 코드
│   └── coco_classes.rs          # COCO 클래스 정의
├── Cargo.toml                   # Rust 프로젝트 설정
└── README.md                    # 이 파일
```

## 🔧 주요 컴포넌트

### 1. 객체 검출 엔진 (`src/lib.rs`)
- `detect_objects()`: 메인 검출 함수
- `preprocess_image()`: 이미지 전처리 (560x560 리사이즈, 정규화)
- `parse_rf_detr_outputs()`: 모델 출력 파싱
- `draw_detections()`: 바운딩 박스 그리기

### 2. COCO 클래스 시스템 (`src/coco_classes.rs`)
- 80개 COCO 객체 클래스 정의
- 클래스 ID와 이름 매핑
- 신뢰도 기반 필터링

### 3. 이미지 처리
- 다양한 이미지 형식 지원 (PNG, JPG, JPEG, BMP, WebP)
- HWC → CHW 변환
- 픽셀 값 정규화 (0-255 → 0-1)

## 📊 출력 예시

```
RF-DETR Object Detection
Loading sample image...

=== Detection Results ===
Found 2 objects:
1. bicycle (Confidence: 92.8%, BBox: [0.003, 0.071, 0.483, 0.992])
2. bicycle (Confidence: 91.1%, BBox: [0.446, 0.049, 0.947, 0.988])

Processed image saved as 'output_with_detections.png'
```

## 🎯 성능 최적화

- **Release 빌드**: `cargo run --release`로 최적화된 실행
- **LTO (Link Time Optimization)**: 전체 프로그램 최적화
- **단일 코드 유닛**: 컴파일 시간 최적화
- **패닉 중단**: 오류 처리 최적화

## 🔍 기술 스택

- **Rust**: 시스템 프로그래밍 언어
- **ONNX Runtime**: 크로스 플랫폼 추론 엔진
- **image**: 이미지 처리 라이브러리
- **imageproc**: 이미지 처리 유틸리티
- **ndarray**: N차원 배열 처리
- **anyhow**: 에러 처리

## 🚧 제한사항

- 현재 Windows 10/11에서만 테스트됨
- GUI 버전은 개발 중 (Dioxus API 호환성 문제)
- GPU 가속 미지원 (CPU 전용)

## 🔮 향후 계획

- [ ] GUI 인터페이스 추가 (Tauri 또는 Egui)
- [ ] GPU 가속 지원
- [ ] 실시간 비디오 처리
- [ ] 웹 인터페이스 (WebAssembly)
- [ ] 더 많은 모델 지원

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요. 