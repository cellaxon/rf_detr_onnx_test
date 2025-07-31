use rf_detr_onnx_test_lib::{detect_objects_with_model, get_model_name, ModelType};
use std::fs;
use std::path::Path;

mod gui;

/// 애플리케이션 실행 모드
#[derive(Debug, Clone, Copy)]
enum RunMode {
    Gui,
    Console,
}

/// 명령행 인자 파싱
fn parse_args() -> Result<(RunMode, Option<String>, Option<ModelType>), String> {
    let args: Vec<String> = std::env::args().collect();

    match args.len() {
        1 => Ok((RunMode::Gui, None, None)),
        2 => Ok((RunMode::Console, Some(args[1].clone()), None)),
        3 => {
            // 이미지 경로와 모델 타입
            let image_path = args[1].clone();
            let model_str = &args[2];
            let model_type = match model_str.to_lowercase().as_str() {
                "original" => Some(ModelType::Original),
                "fp16" => Some(ModelType::FP16),
                "int8" => Some(ModelType::INT8),
                "uint8" => Some(ModelType::UINT8),
                "quantized" => Some(ModelType::Quantized),
                "q4" => Some(ModelType::Q4),
                "q4f16" => Some(ModelType::Q4F16),
                "bnb4" => Some(ModelType::BNB4),
                _ => None,
            };
            
            if model_type.is_none() {
                return Err(format!(
                    "Invalid model type: {}. Available models: original, fp16, int8, uint8, quantized, q4, q4f16, bnb4",
                    model_str
                ));
            }
            
            Ok((RunMode::Console, Some(image_path), model_type))
        }
        _ => Err(format!(
            "Usage:\n  {} [image_path] [model_type]\n  Available models: original, fp16, int8, uint8, quantized, q4, q4f16, bnb4\n  (no arguments: GUI mode)",
            args.get(0).unwrap_or(&"rf_detr_onnx_test".to_string())
        )),
    }
}

/// 콘솔 모드 실행
fn run_console_mode(image_path: &str, model_type: ModelType) -> anyhow::Result<()> {
    // 파일 존재 확인
    if !Path::new(image_path).exists() {
        anyhow::bail!("File not found: {}", image_path);
    }

    println!("RF-DETR Object Detection");
    println!("Model: {}", get_model_name(model_type));
    println!("Loading image: {}", image_path);

    // 이미지 파일 읽기
    let image_data = fs::read(image_path)?;

    // 객체 검출 실행
    let result = detect_objects_with_model(&image_data, model_type)?;

    // 결과 출력
    print_detection_results(&result.detections);
    println!("⏱️ Inference Time: {:.2} ms", result.inference_time_ms);

    // 결과 이미지 저장
    let output_path = "output_with_detections.png";
    result.result_image.save(output_path)?;
    println!("\nProcessed image saved as '{}'", output_path);

    Ok(())
}

/// 검출 결과 출력
fn print_detection_results(detections: &[rf_detr_onnx_test_lib::Detection]) {
    println!("\n=== Detection Results ===");
    println!("Found {} objects:", detections.len());

    if detections.is_empty() {
        println!("No objects detected.");
        return;
    }

    for (i, detection) in detections.iter().enumerate() {
        println!(
            "{}. {} (Confidence: {:.1}%, BBox: [{:.3}, {:.3}, {:.3}, {:.3}])",
            i + 1,
            detection.class,
            detection.confidence * 100.0,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3]
        );
    }
}

fn main() {
    // 명령행 인자 파싱
    let (mode, image_path, model_type) = match parse_args() {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    let result = match mode {
        RunMode::Gui => {
            gui::run_gui();
            Ok(())
        }
        RunMode::Console => {
            let image_path = image_path.unwrap();
            let model_type = model_type.unwrap_or(ModelType::Original);
            run_console_mode(&image_path, model_type)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
