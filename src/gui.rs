use eframe::egui;
use rf_detr_onnx_test_lib::{detect_objects_with_model, get_model_name, Detection, ModelType};
use std::fs;
use std::path::PathBuf;

/// GUI 애플리케이션 실행
pub fn run_gui() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RF-DETR Object Detection",
        options,
        Box::new(|_cc| Ok(Box::new(RfDetrApp::default()))),
    )
    .unwrap();
}

/// RF-DETR GUI 애플리케이션 구조체
struct RfDetrApp {
    detections: Vec<Detection>,
    is_processing: bool,
    error_message: Option<String>,
    selected_image_path: Option<PathBuf>,
    processed_image: Option<egui::TextureHandle>,
    image_size: egui::Vec2,
    inference_time_ms: Option<f64>,
    selected_model: ModelType,
}

impl Default for RfDetrApp {
    fn default() -> Self {
        Self {
            detections: Vec::new(),
            is_processing: false,
            error_message: None,
            selected_image_path: None,
            processed_image: None,
            image_size: egui::Vec2::ZERO,
            inference_time_ms: None,
            selected_model: ModelType::Original,
        }
    }
}

impl eframe::App for RfDetrApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_header(ui);
            self.render_error_message(ui);
            self.render_main_content(ui);
        });
    }
}

impl RfDetrApp {
    /// 헤더 영역 렌더링
    fn render_header(&mut self, ui: &mut egui::Ui) {
        ui.heading("RF-DETR Object Detection");
        ui.add_space(10.0);

        // 모델 선택 드롭다운
        ui.horizontal(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_id_source("model_select")
                .selected_text(get_model_name(self.selected_model))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.selected_model, ModelType::Original, get_model_name(ModelType::Original));
                    ui.selectable_value(&mut self.selected_model, ModelType::FP16, get_model_name(ModelType::FP16));
                    ui.selectable_value(&mut self.selected_model, ModelType::INT8, get_model_name(ModelType::INT8));
                    ui.selectable_value(&mut self.selected_model, ModelType::UINT8, get_model_name(ModelType::UINT8));
                    ui.selectable_value(&mut self.selected_model, ModelType::Quantized, get_model_name(ModelType::Quantized));
                    ui.selectable_value(&mut self.selected_model, ModelType::Q4, get_model_name(ModelType::Q4));
                    ui.selectable_value(&mut self.selected_model, ModelType::Q4F16, get_model_name(ModelType::Q4F16));
                    ui.selectable_value(&mut self.selected_model, ModelType::BNB4, get_model_name(ModelType::BNB4));
                });
        });

        ui.horizontal(|ui| {
            if ui.button("📁 Select Image").clicked() && !self.is_processing {
                self.select_image(ui.ctx());
            }

            if self.is_processing {
                ui.label("Processing...");
            }

            if let Some(path) = &self.selected_image_path {
                ui.label(format!(
                    "Selected: {}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        });

        // 추론 시간 표시
        if let Some(inference_time) = self.inference_time_ms {
            ui.horizontal(|ui| {
                ui.label("⏱️ Inference Time:");
                ui.colored_label(
                    egui::Color32::from_rgb(0, 150, 255),
                    format!("{:.2} ms", inference_time)
                );
            });
        }
    }

    /// 에러 메시지 렌더링
    fn render_error_message(&self, ui: &mut egui::Ui) {
        if let Some(error) = &self.error_message {
            ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
        }
        ui.add_space(10.0);
    }

    /// 메인 콘텐츠 영역 렌더링
    fn render_main_content(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.render_detections_panel(ui);
            ui.separator();
            self.render_image_panel(ui);
        });
    }

    /// 검출 결과 패널 렌더링
    fn render_detections_panel(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.set_min_width(300.0);
            ui.heading(format!("Detections ({})", self.detections.len()));

            if self.detections.is_empty() {
                ui.label("No detections yet. Select an image to get started.");
            } else {
                for (i, detection) in self.detections.iter().enumerate() {
                    self.render_detection_item(ui, i, detection);
                }
            }
        });
    }

    /// 개별 검출 결과 아이템 렌더링
    fn render_detection_item(&self, ui: &mut egui::Ui, index: usize, detection: &Detection) {
        ui.group(|ui| {
            ui.heading(format!("Detection #{}", index + 1));
            ui.label(format!("Class: {}", detection.class));
            ui.label(format!("Confidence: {:.1}%", detection.confidence * 100.0));
            ui.label(format!(
                "BBox: [{:.3}, {:.3}, {:.3}, {:.3}]",
                detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
            ));
        });
    }

    /// 이미지 패널 렌더링
    fn render_image_panel(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.set_min_width(400.0);

            if let Some(texture) = &self.processed_image {
                ui.heading("Processed Image");
                ui.image(texture);
            } else {
                self.render_empty_image_placeholder(ui);
            }
        });
    }

    /// 빈 이미지 플레이스홀더 렌더링
    fn render_empty_image_placeholder(&self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label(egui::RichText::new("📷").size(64.0));
            ui.label("Drag and drop an image here");
            ui.label("or click 'Select Image' to choose a file");
        });
    }

    /// 이미지 파일 선택
    fn select_image(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Image files", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.selected_image_path = Some(path.clone());
            self.process_image(ctx, path);
        }
    }

    /// 이미지 처리
    fn process_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.is_processing = true;
        self.error_message = None;
        self.processed_image = None;
        self.detections.clear();
        self.inference_time_ms = None;

        // 이미지 파일 읽기
        match fs::read(&path) {
            Ok(image_data) => {
                // 객체 검출 실행 (선택된 모델 사용)
                match detect_objects_with_model(&image_data, self.selected_model) {
                    Ok(result) => {
                        self.detections = result.detections;
                        self.inference_time_ms = Some(result.inference_time_ms);
                        self.load_texture(ctx, result.result_image);
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Detection error: {}", e));
                    }
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to read file: {}", e));
            }
        }

        self.is_processing = false;
    }

    /// 텍스처 로딩
    fn load_texture(&mut self, ctx: &egui::Context, result_image: image::RgbImage) {
        let mut buffer = Vec::new();
        if let Ok(()) = result_image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Png,
        ) {
            if let Ok(image) = image::load_from_memory(&buffer) {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as _, rgba.height() as _];

                // ColorImage 생성
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

                let texture = ctx.load_texture("processed_image", color_image, Default::default());
                self.processed_image = Some(texture);
                self.image_size = egui::vec2(size[0] as f32, size[1] as f32);
            }
        }
    }
}
