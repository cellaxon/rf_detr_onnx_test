pub mod coco_classes;

use coco_classes::CocoClass;
use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn};
use ort::{Environment, SessionBuilder, Value};
use std::sync::Arc;

// 상수 정의
const MODEL_INPUT_SIZE: u32 = 560;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const BBOX_COLOR: Rgb<u8> = Rgb([255, 0, 0]); // 빨간색

// 임베디드 리소스
static RF_DETR_BASE_ONNX: &[u8] = include_bytes!("../assets/models/rf-detr-base.onnx");

/// 객체 검출 결과를 나타내는 구조체
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub bbox: [f32; 4], // [x1, y1, x2, y2] in normalized coordinates (0-1)
    pub confidence: f32,
    pub class: CocoClass,
}

/// 시그모이드 함수
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 레터박싱 좌표를 원본 이미지 좌표로 변환
fn letterbox_to_original_coords(
    bbox: [f32; 4], // [x1, y1, x2, y2] in letterboxed coordinates (0-1)
    original_width: u32,
    original_height: u32,
) -> [f32; 4] {
    let aspect_ratio = original_width as f32 / original_height as f32;

    let (scale, offset_x, offset_y) = if aspect_ratio > 1.0 {
        // 가로가 더 긴 경우
        let scale = MODEL_INPUT_SIZE as f32 / original_width as f32;
        let offset_x = 0.0;
        let offset_y = (MODEL_INPUT_SIZE as f32 - MODEL_INPUT_SIZE as f32 / aspect_ratio) / 2.0;
        (scale, offset_x, offset_y)
    } else {
        // 세로가 더 긴 경우
        let scale = MODEL_INPUT_SIZE as f32 / original_height as f32;
        let offset_x = (MODEL_INPUT_SIZE as f32 - MODEL_INPUT_SIZE as f32 * aspect_ratio) / 2.0;
        let offset_y = 0.0;
        (scale, offset_x, offset_y)
    };

    // 레터박싱 좌표를 픽셀 좌표로 변환
    let x1_pixel = bbox[0] * MODEL_INPUT_SIZE as f32;
    let y1_pixel = bbox[1] * MODEL_INPUT_SIZE as f32;
    let x2_pixel = bbox[2] * MODEL_INPUT_SIZE as f32;
    let y2_pixel = bbox[3] * MODEL_INPUT_SIZE as f32;

    // 패딩 제거
    let x1_unpadded = (x1_pixel - offset_x) / scale;
    let y1_unpadded = (y1_pixel - offset_y) / scale;
    let x2_unpadded = (x2_pixel - offset_x) / scale;
    let y2_unpadded = (y2_pixel - offset_y) / scale;

    // 원본 이미지 범위로 클리핑
    let x1_final = x1_unpadded.max(0.0).min(original_width as f32);
    let y1_final = y1_unpadded.max(0.0).min(original_height as f32);
    let x2_final = x2_unpadded.max(0.0).min(original_width as f32);
    let y2_final = y2_unpadded.max(0.0).min(original_height as f32);

    // 정규화된 좌표로 변환 (0-1)
    [
        x1_final / original_width as f32,
        y1_final / original_height as f32,
        x2_final / original_width as f32,
        y2_final / original_height as f32,
    ]
}

/// 이미지 전처리: 리사이징, 레터박싱, 정규화
pub fn preprocess_image(image: &RgbImage) -> anyhow::Result<ArrayD<f32>> {
    let original_width = image.width() as f32;
    let original_height = image.height() as f32;

    // 종횡비 계산
    let aspect_ratio = original_width / original_height;

    let (new_width, new_height, offset_x, offset_y) = if aspect_ratio > 1.0 {
        // 가로가 더 긴 경우
        let new_width = MODEL_INPUT_SIZE as f32;
        let new_height = new_width / aspect_ratio;
        let offset_x = 0.0;
        let offset_y = (MODEL_INPUT_SIZE as f32 - new_height) / 2.0;
        (
            new_width as u32,
            new_height as u32,
            offset_x as u32,
            offset_y as u32,
        )
    } else {
        // 세로가 더 긴 경우
        let new_height = MODEL_INPUT_SIZE as f32;
        let new_width = new_height * aspect_ratio;
        let offset_x = (MODEL_INPUT_SIZE as f32 - new_width) / 2.0;
        let offset_y = 0.0;
        (
            new_width as u32,
            new_height as u32,
            offset_x as u32,
            offset_y as u32,
        )
    };

    // 이미지 리사이즈 (종횡비 유지)
    let resized = image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // 정사각형 캔버스 생성 (회색 배경)
    let mut canvas = RgbImage::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    let padding_color = Rgb([114, 114, 114]); // 회색 패딩

    // 캔버스를 패딩 색상으로 채우기
    for pixel in canvas.pixels_mut() {
        *pixel = padding_color;
    }

    // 리사이즈된 이미지를 캔버스 중앙에 배치
    for y in 0..new_height {
        for x in 0..new_width {
            let canvas_x = x + offset_x;
            let canvas_y = y + offset_y;
            if canvas_x < MODEL_INPUT_SIZE && canvas_y < MODEL_INPUT_SIZE {
                canvas.put_pixel(canvas_x, canvas_y, *resized.get_pixel(x, y));
            }
        }
    }

    // HWC -> CHW 변환 및 정규화 (0~1)
    let mut input_data =
        Vec::with_capacity(1 * 3 * MODEL_INPUT_SIZE as usize * MODEL_INPUT_SIZE as usize);
    for c in 0..3 {
        for y in 0..MODEL_INPUT_SIZE {
            for x in 0..MODEL_INPUT_SIZE {
                let pixel_value = canvas.get_pixel(x, y)[c as usize] as f32 / 255.0;
                input_data.push(pixel_value);
            }
        }
    }

    // 텐서 생성
    Ok(ArrayD::from_shape_vec(
        IxDyn(&[1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize]),
        input_data,
    )?)
}

/// RF-DETR 모델 출력 파싱
pub fn parse_rf_detr_outputs(
    bbox_tensor: &ndarray::ArrayViewD<f32>,  // 바운딩 박스 좌표
    class_tensor: &ndarray::ArrayViewD<f32>, // 클래스 로짓
    original_width: u32,
    original_height: u32,
) -> anyhow::Result<Vec<Detection>> {
    const MAX_DETECTIONS: usize = 100;

    let mut detections = Vec::new();
    let num_queries = class_tensor.shape()[1];

    for q in 0..num_queries.min(MAX_DETECTIONS) {
        // 클래스 확률 계산
        let mut max_conf = 0.0;
        let mut best_class = 0;

        // background 클래스(0) 제외하고 실제 객체 클래스만 처리 (1-80)
        for c in 1..91 {
            let logit = class_tensor[[0, q, c]];
            let conf = sigmoid(logit);
            if conf > max_conf {
                max_conf = conf;
                best_class = c;
            }
        }

        // 신뢰도 임계값 확인
        if max_conf > CONFIDENCE_THRESHOLD {
            // 바운딩 박스 좌표 추출 (레터박싱된 이미지 기준)
            let cx = bbox_tensor[[0, q, 0]];
            let cy = bbox_tensor[[0, q, 1]];
            let w = bbox_tensor[[0, q, 2]];
            let h = bbox_tensor[[0, q, 3]];

            // 유효한 바운딩 박스인지 확인
            if w > 0.0 && h > 0.0 {
                let x1 = (cx - w / 2.0).max(0.0).min(1.0);
                let y1 = (cy - h / 2.0).max(0.0).min(1.0);
                let x2 = (cx + w / 2.0).max(0.0).min(1.0);
                let y2 = (cy + h / 2.0).max(0.0).min(1.0);

                // 레터박싱 좌표를 원본 이미지 좌표로 변환
                let original_bbox =
                    letterbox_to_original_coords([x1, y1, x2, y2], original_width, original_height);

                if let Some(class) = CocoClass::from_id(best_class) {
                    detections.push(Detection {
                        bbox: original_bbox,
                        confidence: max_conf,
                        class,
                    });
                }
            }
        }
    }

    Ok(detections)
}

/// 검출된 객체에 바운딩 박스 그리기
pub fn draw_detections(image: &mut RgbImage, detections: &[Detection]) {
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        let x1 = (x1 * image.width() as f32) as i32;
        let y1 = (y1 * image.height() as f32) as i32;
        let x2 = (x2 * image.width() as f32) as i32;
        let y2 = (y2 * image.height() as f32) as i32;

        let rect = Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32);
        draw_hollow_rect_mut(image, rect, BBOX_COLOR);
    }
}

/// 메인 객체 검출 함수
pub fn detect_objects(image_data: &[u8]) -> anyhow::Result<(Vec<Detection>, RgbImage)> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // ONNX Runtime 환경 생성
    let environment = Arc::new(
        Environment::builder()
            .with_name("rf-detr-embedded")
            .build()?,
    );

    let session = SessionBuilder::new(&environment)?.with_model_from_memory(RF_DETR_BASE_ONNX)?;

    // 이미지 전처리
    let input_array = preprocess_image(&img)?;
    let cow_array = CowArray::from(&input_array);
    let input_value = Value::from_array(session.allocator(), &cow_array)?;

    // 추론 실행
    let outputs = session.run(vec![input_value])?;

    // 결과 파싱
    let mut detections = Vec::new();
    if outputs.len() >= 2 {
        let logits_tensor = outputs[0].try_extract::<f32>()?;
        let boxes_tensor = outputs[1].try_extract::<f32>()?;

        let logits_view = logits_tensor.view();
        let boxes_view = boxes_tensor.view();

        // RF-DETR 출력 파싱 (boxes가 클래스 로짓, logits가 바운딩 박스)
        detections = parse_rf_detr_outputs(&logits_view, &boxes_view, img.width(), img.height())?;
    }

    // 바운딩 박스가 포함된 이미지 생성
    let mut result_image = img.clone();
    draw_detections(&mut result_image, &detections);

    Ok((detections, result_image))
}
