mod coco_classes;

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
static SAMPLE_IMAGE: &[u8] = include_bytes!("../assets/images/sample.png");

#[derive(Debug)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2]
    confidence: f32,
    class: CocoClass,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn load_image() -> anyhow::Result<RgbImage> {
    let img = ImageReader::new(std::io::Cursor::new(SAMPLE_IMAGE))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    Ok(img)
}

fn preprocess_image(image: &RgbImage) -> anyhow::Result<ArrayD<f32>> {
    // 이미지 리사이즈
    let resized = image::imageops::resize(
        image,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );

    // HWC -> CHW 변환 및 정규화 (0~1)
    let mut input_data =
        Vec::with_capacity(1 * 3 * MODEL_INPUT_SIZE as usize * MODEL_INPUT_SIZE as usize);
    for c in 0..3 {
        for y in 0..MODEL_INPUT_SIZE {
            for x in 0..MODEL_INPUT_SIZE {
                let pixel_value = resized.get_pixel(x, y)[c as usize] as f32 / 255.0;
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

fn parse_rf_detr_outputs(
    bbox_tensor: &ndarray::ArrayViewD<f32>,  // 바운딩 박스 좌표
    class_tensor: &ndarray::ArrayViewD<f32>, // 클래스 로짓
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();
    let bbox_shape = bbox_tensor.shape();
    let class_shape = class_tensor.shape();

    // 유효한 텐서 형태 확인
    if class_shape[2] != 91 || bbox_shape[2] != 4 {
        return Ok(detections);
    }

    let num_queries = class_shape[1];

    for q in 0..num_queries {
        // 클래스 확률 계산
        let mut max_conf = 0.0;
        let mut best_class = 0;

        // background 클래스 제외하고 처리
        for c in 0..90 {
            let logit = class_tensor[[0, q, c]];
            let conf = sigmoid(logit);
            if conf > max_conf {
                max_conf = conf;
                best_class = c;
            }
        }

        // 신뢰도 임계값 확인
        if max_conf > CONFIDENCE_THRESHOLD {
            // 바운딩 박스 좌표 추출
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

                if let Some(class) = CocoClass::from_id(best_class) {
                    detections.push(Detection {
                        bbox: [x1, y1, x2, y2],
                        confidence: max_conf,
                        class,
                    });
                }
            }
        }
    }

    Ok(detections)
}

fn draw_detections(image: &mut RgbImage, detections: &[Detection]) {
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

fn print_detections(detections: &[Detection]) {
    println!("\n=== 바운딩 박스 검출 결과 ===");
    println!("총 검출된 객체 수: {}", detections.len());

    for (i, detection) in detections.iter().enumerate() {
        println!(
            "객체 {}: 클래스={} (ID: {}), 신뢰도={:.1}%, 바운딩박스=[{:.3}, {:.3}, {:.3}, {:.3}]",
            i + 1,
            detection.class,
            detection.class.id(),
            detection.confidence * 100.0,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3]
        );
    }
    println!("=============================\n");
}

fn save_results(original_image: &RgbImage, detections: &[Detection]) -> anyhow::Result<()> {
    // 바운딩 박스가 포함된 이미지 저장
    let mut result_image = original_image.clone();
    draw_detections(&mut result_image, detections);
    result_image.save("output_with_detections.png")?;
    println!("바운딩 박스가 포함된 결과 이미지가 'output_with_detections.png'로 저장되었습니다.");

    // 원본 이미지 저장
    original_image.save("output_image.png")?;
    println!("원본 이미지가 'output_image.png'로 저장되었습니다.");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("RF-DETR 모델 로딩 중...");
    let environment = Arc::new(
        Environment::builder()
            .with_name("rf-detr-embedded")
            .build()?,
    );

    let session = SessionBuilder::new(&environment)?.with_model_from_memory(RF_DETR_BASE_ONNX)?;

    println!("이미지 로딩 중...");
    let image = load_image()?;

    println!("이미지 전처리 중...");
    let input_array = preprocess_image(&image)?;
    let cow_array = CowArray::from(&input_array);
    let input_value = Value::from_array(session.allocator(), &cow_array)?;

    println!("객체 검출 실행 중...");
    let outputs = session.run(vec![input_value])?;

    // 결과 파싱
    let mut detections = Vec::new();
    if outputs.len() >= 2 {
        let logits_tensor = outputs[0].try_extract::<f32>()?;
        let boxes_tensor = outputs[1].try_extract::<f32>()?;

        let logits_view = logits_tensor.view();
        let boxes_view = boxes_tensor.view();

        // RF-DETR 출력 파싱 (boxes가 클래스 로짓, logits가 바운딩 박스)
        detections = parse_rf_detr_outputs(&logits_view, &boxes_view)?;
    }

    // 결과 출력
    print_detections(&detections);

    // 결과 저장
    save_results(&image, &detections)?;

    println!("객체 검출 완료!");
    Ok(())
}
