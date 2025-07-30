# RF-DETR ONNX Test

RF-DETR (Real-time Detection Transformer) 객체 검출 모델을 Rust로 구현한 프로젝트입니다. ONNX Runtime을 사용하여 모델 추론을 수행하고, 콘솔 및 GUI 인터페이스를 제공합니다.

## 주요 기능

- **ONNX 모델 추론**: RF-DETR 모델을 사용한 실시간 객체 검출
- **이미지 전처리**: 자동 이미지 리사이징 및 정규화
- **바운딩 박스 시각화**: 검출된 객체에 대한 바운딩 박스 및 클래스 정보 표시
- **다중 인터페이스**: 콘솔 모드와 GUI 모드 지원
- **드래그 앤 드롭**: GUI에서 이미지 파일 선택 지원

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

#### 콘솔 모드 (기본)
```bash
cargo run
```

#### GUI 모드
```bash
cargo run -- --gui
```

## 프로젝트 구조

```
rf_detr_onnx_test/
├── src/
│   ├── main.rs          # 메인 실행 파일 (콘솔/GUI 모드 선택)
│   ├── lib.rs           # 핵심 라이브러리 (ONNX 추론, 이미지 처리)
│   ├── gui.rs           # egui 기반 GUI 구현
│   └── coco_classes.rs  # COCO 클래스 정의
├── assets/
│   ├── models/
│   │   └── rf-detr-base.onnx  # ONNX 모델 파일
│   ├── images/
│   │   └── sample.png         # 샘플 이미지
│   └── styles.css             # GUI 스타일 (사용되지 않음)
├── Cargo.toml
└── README.md
```

## 사용된 기술

### 핵심 라이브러리
- **ort**: ONNX Runtime Rust 바인딩
- **image**: 이미지 처리 및 변환
- **ndarray**: 다차원 배열 연산
- **imageproc**: 이미지 처리 및 그리기

### GUI 라이브러리
- **egui**: 즉시 모드 GUI 프레임워크
- **eframe**: egui 애플리케이션 프레임워크
- **rfd**: 파일 다이얼로그

### 기타
- **anyhow**: 에러 처리
- **rusttype**: 폰트 렌더링 (텍스트 그리기용)

## 기능 상세

### 객체 검출
- RF-DETR 모델을 사용한 80개 COCO 클래스 검출
- 신뢰도 점수 기반 필터링
- 바운딩 박스 좌표 추출 및 변환

### 이미지 처리
- 자동 560x560 리사이징
- HWC → CHW 변환
- 픽셀 값 정규화 (0-255 → 0-1)
- 바운딩 박스 및 클래스 정보 시각화

### GUI 기능
- 직관적인 파일 선택 인터페이스
- 실시간 이미지 처리 및 표시
- 검출 결과 목록 표시
- 에러 메시지 표시

## 개발 과정

### 초기 구현
- ONNX Runtime을 사용한 기본 추론 구현
- 이미지 전처리 및 후처리 로직 구현
- 바운딩 박스 파싱 및 시각화

### GUI 구현
- **Dioxus 시도**: API 호환성 문제로 실패
- **egui 채택**: 안정적이고 성숙한 GUI 프레임워크
- 파일 선택 및 이미지 표시 기능 구현

### 최적화
- 코드 리팩토링 및 모듈화
- 에러 처리 개선
- 사용자 인터페이스 개선

## 문제 해결

### ONNX Runtime 관련
- `0xc000007b` 오류: ort 크레이트로 해결
- 텐서 형식 불일치: 올바른 전처리 파이프라인 구현

### GUI 관련
- Dioxus API 호환성 문제: egui로 전환
- 이미지 텍스처 로딩: ColorImage API 사용

## 향후 계획

- [ ] 드래그 앤 드롭 기능 완성
- [ ] 배치 처리 지원
- [ ] 실시간 비디오 처리
- [ ] 모델 성능 최적화
- [ ] 추가 GUI 기능 (설정, 결과 저장 등)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

---

**참고**: 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 프로덕션 환경에서 사용하기 전에 충분한 테스트를 권장합니다. 