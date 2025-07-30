use rf_detr_onnx_test_lib::detect_objects;
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
fn parse_args() -> Result<RunMode, String> {
    let args: Vec<String> = std::env::args().collect();

    match args.len() {
        1 => Ok(RunMode::Gui),
        2 => Ok(RunMode::Console),
        _ => Err(format!(
            "Usage:\n  {} [image_path]\n  (no arguments: GUI mode)",
            args.get(0).unwrap_or(&"rf_detr_onnx_test".to_string())
        )),
    }
}

/// 콘솔 모드 실행
fn run_console_mode(image_path: &str) -> anyhow::Result<()> {
    // 파일 존재 확인
    if !Path::new(image_path).exists() {
        anyhow::bail!("File not found: {}", image_path);
    }

    println!("RF-DETR Object Detection");
    println!("Loading image: {}", image_path);

    // 이미지 파일 읽기
    let image_data = fs::read(image_path)?;

    // 객체 검출 실행
    let (detections, result_image) = detect_objects(&image_data)?;

    // 결과 출력
    print_detection_results(&detections);

    // 결과 이미지 저장
    let output_path = "output_with_detections.png";
    result_image.save(output_path)?;
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
    let mode = match parse_args() {
        Ok(mode) => mode,
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
            let args: Vec<String> = std::env::args().collect();
            let image_path = &args[1];
            run_console_mode(image_path)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
