use image::ImageReader;
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

    // 8. 출력 확인
    println!("ONNX outputs: {:?}", outputs);

    Ok(())
}
