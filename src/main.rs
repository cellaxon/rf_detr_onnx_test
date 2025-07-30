use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn};
use ort::{Environment, SessionBuilder, Value};
use std::sync::Arc;

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

    // 8. 출력 결과 처리 및 시각화
    println!("ONNX outputs: {:?}", outputs);
    println!("출력 텐서 개수: {}", outputs.len());

    // --- 핵심: 실제 모델 출력으로부터 바운딩 박스 추출 ---
    let mut detections = Vec::new();
    if let Some(output) = outputs.get(0) {
        if let Ok(array) = output.try_extract::<f32>() {
            let array_view = array.view();
            detections = process_detections(&array_view, 0.5)?; // 신뢰도 0.5
        }
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

    // (이하 draw_detections 및 저장 부분은 동일)
    let mut result_image = img.clone();
    draw_detections(&mut result_image, &detections);

    result_image.save("output_with_detections.png")?;
    println!("바운딩 박스가 포함된 결과 이미지가 'output_with_detections.png'로 저장되었습니다.");

    img.save("output_image.png")?;
    println!("원본 이미지가 'output_image.png'로 저장되었습니다.");

    Ok(())
}

// === 이하 Detection, process_detections, draw_detections 정의는 기존과 동일 ===

#[derive(Debug)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2]
    confidence: f32,
    class_id: usize,
}

fn process_detections(
    output: &ndarray::ArrayViewD<f32>,
    confidence_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();

    let shape = output.shape();
    println!("Output shape for processing: {:?}", shape);

    if shape.len() >= 3 {
        let num_queries = shape[1];

        // 출력이 [1, 300, 4] 형태라면 바운딩 박스 좌표만 있는 것
        if shape[2] == 4 {
            for query_idx in 0..num_queries {
                let bbox = [
                    output[[0, query_idx, 0]], // x1
                    output[[0, query_idx, 1]], // y1
                    output[[0, query_idx, 2]], // x2
                    output[[0, query_idx, 3]], // y2
                ];

                // 바운딩 박스가 유효한지 확인 (0이 아닌 경우)
                if bbox[0] != 0.0 || bbox[1] != 0.0 || bbox[2] != 0.0 || bbox[3] != 0.0 {
                    detections.push(Detection {
                        bbox,
                        confidence: 1.0, // 임시로 1.0 설정
                        class_id: 0,     // 임시로 0 설정
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

        // 스케일링
        let x1 = (x1 * image.width() as f32) as i32;
        let y1 = (y1 * image.height() as f32) as i32;
        let x2 = (x2 * image.width() as f32) as i32;
        let y2 = (y2 * image.height() as f32) as i32;

        let rect = Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32);
        draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
        println!(
            "Detection: class={}, conf={:.3}, bbox=[{:.3}, {:.3}, {:.3}, {:.3}]",
            detection.class_id, detection.confidence, x1, y1, x2, y2
        );
    }
}
