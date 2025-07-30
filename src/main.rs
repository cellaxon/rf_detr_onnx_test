use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn, s};
use ort::{Environment, SessionBuilder, Value};
use std::sync::Arc;

/// softmax 함수(로짓 벡터 입력, 최대값 반환)
fn softmax_max(logits: &[f32]) -> (usize, f32) {
    // overflow 방지용
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();
    let (idx, &prob) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (idx, prob)
}

static RF_DETR_BASE_ONNX: &[u8] = include_bytes!("../assets/models/rf-detr-base.onnx");
static SAMPLE_IMAGE: &[u8] = include_bytes!("../assets/images/sample.png");

fn main() -> anyhow::Result<()> {
    // 1. ONNX Runtime 환경 생성
    let environment = Arc::new(
        Environment::builder()
            .with_name("rf-detr-embedded")
            .build()?,
    );

    // 2. 메모리에서 모델 로드하여 세션 생성
    let session = SessionBuilder::new(&environment)?.with_model_from_memory(RF_DETR_BASE_ONNX)?;

    // 3. 이미지 불러오기 및 전처리
    let img = ImageReader::new(std::io::Cursor::new(SAMPLE_IMAGE))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let resized = image::imageops::resize(&img, 560, 560, image::imageops::FilterType::Triangle);

    // 4. HWC -> CHW 변환, 0~1 정규화
    let mut input_data = Vec::with_capacity(1 * 3 * 560 * 560);
    for c in 0..3 {
        for y in 0..560 {
            for x in 0..560 {
                let pixel_value = resized.get_pixel(x, y)[c as usize] as f32 / 255.0;
                input_data.push(pixel_value);
            }
        }
    }
    // 5. ArrayD로 직접 생성
    let input_array = ArrayD::from_shape_vec(IxDyn(&[1, 3, 560, 560]), input_data)?;

    // 6. 배열 참조를 직접 CowArray로 변환
    let cow_array = CowArray::from(&input_array);
    let input_value = Value::from_array(session.allocator(), &cow_array)?;

    // 7. 추론 실행
    let outputs = session.run(vec![input_value])?;
    println!("\nONNX outputs len: {}", outputs.len());

    // --- 올바른 바운딩 박스/클래스 파싱 ---
    let mut detections = Vec::new();
    if outputs.len() >= 2 {
        let logits_tensor = outputs.get(0).unwrap().try_extract::<f32>()?; // pred_logits
        let boxes_tensor = outputs.get(1).unwrap().try_extract::<f32>()?; // pred_boxes

        let logits_view = logits_tensor.view();
        let boxes_view = boxes_tensor.view();

        // 디버깅: 출력 텐서 형태와 값 확인
        println!("Logits shape: {:?}", logits_view.shape());
        println!("Boxes shape: {:?}", boxes_view.shape());

        // 첫 번째 쿼리의 값들 확인
        if logits_view.shape()[1] > 0 {
            println!(
                "First query logits: {:?}",
                logits_view.slice(s![0, 0, ..]).to_vec()
            );
            println!(
                "First query boxes (first 10): {:?}",
                boxes_view.slice(s![0, 0, ..10]).to_vec()
            );
        }

        // 텐서 역할을 바꿔서 처리 (boxes가 실제로는 logits일 수 있음)
        detections = parse_rf_detr_outputs_alternative(&logits_view, &boxes_view, 0.5)?;
    }

    // 바운딩 박스 정보 텍스트 출력
    println!("\n=== 바운딩 박스 검출 결과 ===");
    println!("총 검출된 객체 수: {}", detections.len());
    for (i, detection) in detections.iter().enumerate() {
        println!(
            "객체 {}: 클래스={}, 신뢰도={:.3}, 바운딩박스=[{:.3}, {:.3}, {:.3}, {:.3}]",
            i + 1,
            detection.class_id,
            detection.confidence,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3]
        );
    }
    println!("=============================\n");

    // 바운딩 박스 시각화
    let mut result_image = img.clone();
    draw_detections(&mut result_image, &detections);

    result_image.save("output_with_detections.png")?;
    println!("바운딩 박스가 포함된 결과 이미지가 'output_with_detections.png'로 저장되었습니다.");

    img.save("output_image.png")?;
    println!("원본 이미지가 'output_image.png'로 저장되었습니다.");

    Ok(())
}

#[derive(Debug)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2]
    confidence: f32,
    class_id: usize,
}
fn parse_rf_detr_outputs(
    logits: &ndarray::ArrayViewD<f32>,
    boxes: &ndarray::ArrayViewD<f32>,
    threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut result = Vec::new();
    let shape = logits.shape();
    let num_queries = shape[1];
    let num_classes = shape[2];

    println!(
        "Processing {} queries with {} classes",
        num_queries, num_classes
    );

    for q in 0..num_queries {
        let mut max_conf = 0.0;
        let mut best_class = 0;

        // 마지막 클래스(background) 제외하고 처리
        for c in 0..(num_classes - 1) {
            let logit = logits[[0, q, c]];
            let conf = sigmoid(logit);
            if conf > max_conf {
                max_conf = conf;
                best_class = c;
            }
        }

        if max_conf > threshold {
            let cx = boxes[[0, q, 0]];
            let cy = boxes[[0, q, 1]];
            let w = boxes[[0, q, 2]];
            let h = boxes[[0, q, 3]];

            // 유효한 바운딩 박스인지 확인
            if w > 0.0 && h > 0.0 && cx >= 0.0 && cy >= 0.0 && cx <= 1.0 && cy <= 1.0 {
                let x1 = (cx - w / 2.0).max(0.0).min(1.0);
                let y1 = (cy - h / 2.0).max(0.0).min(1.0);
                let x2 = (cx + w / 2.0).max(0.0).min(1.0);
                let y2 = (cy + h / 2.0).max(0.0).min(1.0);

                result.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    confidence: max_conf,
                    class_id: best_class,
                });

                println!(
                    "Detection found: class={}, conf={:.3}, bbox=[{:.3}, {:.3}, {:.3}, {:.3}]",
                    best_class, max_conf, x1, y1, x2, y2
                );
            }
        }
    }

    println!("Total detections: {}", result.len());
    Ok(result)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn parse_rf_detr_outputs_alternative(
    logits: &ndarray::ArrayViewD<f32>,
    boxes: &ndarray::ArrayViewD<f32>,
    threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut result = Vec::new();
    let logits_shape = logits.shape();
    let boxes_shape = boxes.shape();

    println!(
        "Alternative parsing - Logits: {:?}, Boxes: {:?}",
        logits_shape, boxes_shape
    );

    // boxes 텐서가 91개 값을 가지므로, 이는 클래스 로짓일 가능성이 높음
    if boxes_shape[2] == 91 {
        let num_queries = boxes_shape[1];
        let num_classes = boxes_shape[2];

        for q in 0..num_queries {
            let mut max_conf = 0.0;
            let mut best_class = 0;

            // background 클래스 제외하고 처리
            for c in 0..(num_classes - 1) {
                let logit = boxes[[0, q, c]];
                let conf = sigmoid(logit);
                if conf > max_conf {
                    max_conf = conf;
                    best_class = c;
                }
            }

            if max_conf > threshold {
                // logits 텐서에서 바운딩 박스 좌표 추출 (4개 값)
                if logits_shape[2] == 4 {
                    let cx = logits[[0, q, 0]];
                    let cy = logits[[0, q, 1]];
                    let w = logits[[0, q, 2]];
                    let h = logits[[0, q, 3]];

                    // 유효한 바운딩 박스인지 확인
                    if w > 0.0 && h > 0.0 {
                        let x1 = (cx - w / 2.0).max(0.0).min(1.0);
                        let y1 = (cy - h / 2.0).max(0.0).min(1.0);
                        let x2 = (cx + w / 2.0).max(0.0).min(1.0);
                        let y2 = (cy + h / 2.0).max(0.0).min(1.0);

                        result.push(Detection {
                            bbox: [x1, y1, x2, y2],
                            confidence: max_conf,
                            class_id: best_class,
                        });

                        println!(
                            "Alt Detection: class={}, conf={:.3}, bbox=[{:.3}, {:.3}, {:.3}, {:.3}]",
                            best_class, max_conf, x1, y1, x2, y2
                        );
                    }
                }
            }
        }
    }

    println!("Alternative total detections: {}", result.len());
    Ok(result)
}

fn draw_detections(image: &mut RgbImage, detections: &[Detection]) {
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        let x1 = (x1 * image.width() as f32) as i32;
        let y1 = (y1 * image.height() as f32) as i32;
        let x2 = (x2 * image.width() as f32) as i32;
        let y2 = (y2 * image.height() as f32) as i32;
        let rect = Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32);
        draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
        println!(
            "Detection: class={}, conf={:.3}, bbox=[{}, {}, {}, {}]",
            detection.class_id, detection.confidence, x1, y1, x2, y2
        );
    }
}
