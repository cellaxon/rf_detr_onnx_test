use rf_detr_onnx_test_lib::detect_objects;
use std::fs;
use std::path::Path;

mod gui;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    match args.len() {
        1 => {
            // 인자가 없으면 GUI 모드
            gui::run_gui();
            Ok(())
        }
        2 => {
            // 인자가 1개(프로그램명 + 이미지 경로)면 해당 이미지로 콘솔 모드
            let image_path = &args[1];
            if !Path::new(image_path).exists() {
                eprintln!("File not found: {}", image_path);
                std::process::exit(1);
            }
            let image_data = fs::read(image_path)?;
            run_console_mode_with_image(&image_data)
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  {} [image_path]", args[0]);
            eprintln!("  (no arguments: GUI mode)");
            std::process::exit(1);
        }
    }
}

fn run_console_mode_with_image(image_data: &[u8]) -> anyhow::Result<()> {
    println!("RF-DETR Object Detection");
    println!("Loading image...");

    // 객체 검출 실행
    let (detections, result_image) = detect_objects(image_data)?;

    // 결과 출력
    println!("\n=== Detection Results ===");
    println!("Found {} objects:", detections.len());

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

    // 결과 이미지 저장
    result_image.save("output_with_detections.png")?;
    println!("\nProcessed image saved as 'output_with_detections.png'");

    Ok(())
}
