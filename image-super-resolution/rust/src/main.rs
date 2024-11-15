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

const INTERMEDIATE_FEATURES: usize = 15;

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
    ImageReader::open(img_path)?.decode()
}

fn load_img_as_tensor<B: Backend>(img_path: &Path, dev: &B::Device) -> ImageResult<Tensor<B, 3>> {
    let img = load_img(img_path)?;
    let w = img.width() as usize;
    let h = img.height() as usize;
    let c = img.color().channel_count() as usize;
    let bytes = img.into_rgb32f().into_vec();

    let flat_tensor: Tensor<B, 1> = Tensor::from_floats(&bytes[..], dev);

    // FIXME Check the data layout, we're assuming a lot here
    Ok(flat_tensor.reshape([w, h, c]))
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
                for entry in WalkDir::new(small_dir) {
                    let entry = entry?;
                    let small_path = entry.path();
                    let base = small_path.components().last().unwrap();

                    let large_path = {
                        let mut p = large_dir.clone();
                        p.push(base);
                        p
                    };

                    let small = load_img_as_tensor::<B>(small_path, &dev);

                    let large = load_img_as_tensor::<B>(&large_path, &dev);
                }
            }
        }
    }

    Ok(())
}
