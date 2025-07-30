use eframe::egui;
use rf_detr_onnx_test_lib::{detect_objects, Detection};
use std::fs;
use std::path::PathBuf;

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

struct RfDetrApp {
    detections: Vec<Detection>,
    is_processing: bool,
    error_message: Option<String>,
    selected_image_path: Option<PathBuf>,
    processed_image: Option<egui::TextureHandle>,
    image_size: egui::Vec2,
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
        }
    }
}

impl eframe::App for RfDetrApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("RF-DETR Object Detection");
            ui.add_space(10.0);

            // Top controls
            ui.horizontal(|ui| {
                if ui.button("📁 Select Image").clicked() && !self.is_processing {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Image files", &["png", "jpg", "jpeg", "bmp", "webp"])
                        .pick_file()
                    {
                        self.selected_image_path = Some(path.clone());
                        self.process_image(ctx, path);
                    }
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

            // Error message
            if let Some(error) = &self.error_message {
                ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
            }

            ui.add_space(10.0);

            // Main content
            ui.horizontal(|ui| {
                // Left panel - Detections
                ui.vertical(|ui| {
                    ui.set_min_width(300.0);
                    ui.heading(format!("Detections ({})", self.detections.len()));

                    if self.detections.is_empty() {
                        ui.label("No detections yet. Select an image to get started.");
                    } else {
                        for (i, detection) in self.detections.iter().enumerate() {
                            ui.group(|ui| {
                                ui.heading(format!("Detection #{}", i + 1));
                                ui.label(format!("Class: {}", detection.class));
                                ui.label(format!(
                                    "Confidence: {:.1}%",
                                    detection.confidence * 100.0
                                ));
                                ui.label(format!(
                                    "BBox: [{:.3}, {:.3}, {:.3}, {:.3}]",
                                    detection.bbox[0],
                                    detection.bbox[1],
                                    detection.bbox[2],
                                    detection.bbox[3]
                                ));
                            });
                        }
                    }
                });

                ui.separator();

                // Right panel - Image display
                ui.vertical(|ui| {
                    ui.set_min_width(400.0);

                    if let Some(texture) = &self.processed_image {
                        ui.heading("Processed Image");
                        ui.image(texture);
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(100.0);
                            ui.label(egui::RichText::new("📷").size(64.0));
                            ui.label("Drag and drop an image here");
                            ui.label("or click 'Select Image' to choose a file");
                        });
                    }
                });
            });
        });
    }
}

impl RfDetrApp {
    fn process_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.is_processing = true;
        self.error_message = None;
        self.processed_image = None;
        self.detections.clear();

        // Process the image
        match fs::read(&path) {
            Ok(image_data) => {
                match detect_objects(&image_data) {
                    Ok((detections, result_image)) => {
                        self.detections = detections;

                        // Convert the processed image to egui texture
                        let mut buffer = Vec::new();
                        if let Ok(()) = result_image.write_to(
                            &mut std::io::Cursor::new(&mut buffer),
                            image::ImageFormat::Png,
                        ) {
                            if let Ok(image) = image::load_from_memory(&buffer) {
                                let rgba = image.to_rgba8();
                                let size = [rgba.width() as _, rgba.height() as _];

                                // Create ColorImage from raw RGBA data
                                let color_image =
                                    egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

                                let texture = ctx.load_texture(
                                    "processed_image",
                                    color_image,
                                    Default::default(),
                                );
                                self.processed_image = Some(texture);
                                self.image_size = egui::vec2(size[0] as f32, size[1] as f32);
                            }
                        }
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Error: {}", e));
                    }
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to read file: {}", e));
            }
        }

        self.is_processing = false;
    }
}
