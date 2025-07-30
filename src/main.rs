use rf_detr_onnx_test::{Detection, detect_objects};

mod gui;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--gui" {
        gui::run_gui();
        Ok(())
    } else {
        run_console_mode()
    }
}

fn run_console_mode() -> anyhow::Result<()> {
    println!("RF-DETR Object Detection");
    println!("Loading sample image...");

    // 샘플 이미지 데이터 (임베디드)
    let sample_image_data = include_bytes!("../assets/images/sample.png");

    // 객체 검출 실행
    let (detections, result_image) = detect_objects(sample_image_data)?;

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
    println!("\nTo run with GUI, use: cargo run -- --gui");

    Ok(())
}
