use std::path::{Path, PathBuf};

use burn::{
    backend::Wgpu,
    nn::{conv::Conv2d, pool::AdaptiveAvgPool2d, Dropout, Linear, Relu},
    prelude::*,
};
use clap::{Parser, Subcommand};
use image::{DynamicImage, ImageReader, ImageResult};
use nn::{conv::Conv2dConfig, pool::AdaptiveAvgPool2dConfig, DropoutConfig, LinearConfig};

use walkdir::WalkDir;

type fX = f32;

const W_SMALL: usize = 700;
const H_SMALL: usize = 700;
const W_LARGE: usize = 1400;
const H_LARGE: usize = 1400;
const C: usize = 3;
const INTERMEDIATE_FEATURES: usize = 15;

const TRAIN_BATCH: usize = 10;
const VALID_BATCH: usize = 10;

#[derive(Parser)]
#[command(version, about = "Image super-resolution trainer", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        small_dir: PathBuf,
        large_dir: PathBuf,
        output_dir: PathBuf,

        #[arg(long, default_value_t = 1e-4)]
        lr: fX,

        #[arg(long, default_value_t = 100)]
        epochs: usize,

        /// Moving average window for reporting
        #[arg(long, default_value_t = 100)]
        recent_window: usize,
    },
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    deep: Conv2d<B>,
    deeper: Conv2d<B>,
    deepest: Conv2d<B>,
    // pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    // linear1: Linear<B>,
    // linear2: Linear<B>,
    activation: Relu,
}
#[derive(Config, Debug)]
pub struct ModelConfig {
    // num_classes: usize,
    // hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            deep: Conv2dConfig::new([3, INTERMEDIATE_FEATURES], [7, 7]).init(device),
            deeper: Conv2dConfig::new([INTERMEDIATE_FEATURES, INTERMEDIATE_FEATURES], [5, 5])
                .init(device),
            deepest: Conv2dConfig::new([INTERMEDIATE_FEATURES, 3], [3, 3]).init(device),
            // pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            // linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            // linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

fn load_img(img_path: &Path) -> ImageResult<DynamicImage> {
    ImageReader::open(img_path)?.with_guessed_format()?.decode()
}

/// If the image is smaller than the specified minimums, Ok(None) is returned
/// Otherwise, the image cropped to the minimums is returned
fn load_cropped_img_as_tensor<B: Backend>(
    img_path: &Path,
    dev: &B::Device,
    min_width: usize,
    min_height: usize,
) -> ImageResult<Option<Tensor<B, 3>>> {
    let img = load_img(img_path)?;
    let w = img.width() as usize;
    let h = img.height() as usize;

    if w < min_width || h < min_height {
        return Ok(None);
    }

    let bytes = img.into_rgb32f().into_vec();

    let flat_tensor: Tensor<B, 1> = Tensor::from_floats(&bytes[..], dev);

    // We know this because of the into_rgb32f() call which forces it to RGB
    let c = 3usize;

    // The data layout appears to be (w, h, c), see ImageBuffer::pixel_indices_unchecked
    Ok(Some(flat_tensor.reshape([w, h, c]).slice([
        Some((0i64, min_width as i64)),
        Some((0i64, min_height as i64)),
        None,
    ])))
}

#[derive(Copy, Clone)]
enum DatumType {
    Train,
    Test,
    Valid,
}

fn datum_type(img_path: &Path) -> DatumType {
    let last = img_path.components().last().unwrap();
    let s = last.as_os_str().to_string_lossy();

    let c = s.chars().last().unwrap();

    let s_c = c.to_string();

    let h = u8::from_str_radix(s_c.as_str(), 16).unwrap();

    println!("{} -> {}", img_path.display(), h);

    match h {
        0 => DatumType::Test,
        1..=9 => DatumType::Valid,
        10..=15 => DatumType::Train,
        _ => panic!(),
    }
}

fn main() -> Result<(), walkdir::Error> {
    let cli = Cli::parse();

    type B = Wgpu<f32, i32>;
    let dev = Default::default();

    match &cli.command {
        Commands::Train {
            small_dir,
            large_dir,
            output_dir,
            lr,
            epochs,
            recent_window,
        } => {
            for epoch in 0..*epochs {
                let mut train_batch_small: Vec<Tensor<B, 3>> = Vec::with_capacity(TRAIN_BATCH);
                let mut train_batch_large: Vec<Tensor<B, 3>> = Vec::with_capacity(TRAIN_BATCH);
                let mut valid_batch_small: Vec<Tensor<B, 3>> = Vec::with_capacity(VALID_BATCH);
                let mut valid_batch_large: Vec<Tensor<B, 3>> = Vec::with_capacity(VALID_BATCH);
                for entry in WalkDir::new(small_dir) {
                    let entry = entry?;
                    if entry.path().is_dir() {
                        continue;
                    }
                    let small_path = entry.path();
                    let base = small_path.components().last().unwrap();

                    let large_path = {
                        let mut p = large_dir.clone();
                        p.push(base);
                        p
                    };

                    println!("small: {}", small_path.display());
                    println!("large: {}", large_path.display());

                    let small = load_cropped_img_as_tensor::<B>(small_path, &dev, W_SMALL, H_SMALL)
                        .unwrap();

                    if let Some(small) = small {
                        let large =
                            load_cropped_img_as_tensor::<B>(&large_path, &dev, W_LARGE, H_LARGE)
                                .unwrap();

                        if let Some(large) = large {
                            let type_ = datum_type(small_path);

                            match type_ {
                                DatumType::Train => {
                                    train_batch_small.push(small);
                                    train_batch_large.push(large);

                                    if train_batch_small.len() >= TRAIN_BATCH {
                                        let small = Tensor::stack::<4>(train_batch_small, 0);
                                        let large = Tensor::stack::<4>(train_batch_large, 0);
                                        train_batch_small = Vec::with_capacity(TRAIN_BATCH);
                                        train_batch_large = Vec::with_capacity(TRAIN_BATCH);

                                        println!("train batch");
                                    }
                                }
                                DatumType::Valid => {
                                    valid_batch_small.push(small);
                                    valid_batch_large.push(large);

                                    if valid_batch_small.len() >= VALID_BATCH {
                                        let small = Tensor::stack::<4>(valid_batch_small, 0);
                                        let large = Tensor::stack::<4>(valid_batch_large, 0);
                                        valid_batch_small = Vec::with_capacity(VALID_BATCH);
                                        valid_batch_large = Vec::with_capacity(VALID_BATCH);

                                        println!("valid batch");
                                    }
                                }
                                DatumType::Test => {
                                    // do nothing
                                }
                            }
                        } else {
                            println!("\tlarge too small");
                        }
                    } else {
                        println!("\tsmall too small");
                    }
                }
            }
        }
    }

    Ok(())
}
