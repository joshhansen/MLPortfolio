use std::{
    io::{stdout, Write},
    path::{Path, PathBuf},
};

use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    module::AutodiffModule,
    nn::{conv::Conv2d, loss::Reduction, Dropout},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
    tensor::{backend::AutodiffBackend, cast::ToElement},
};
use clap::{Parser, Subcommand};
use image::{
    imageops::{resize, FilterType},
    DynamicImage, ImageReader, ImageResult,
};
use nn::{conv::Conv2dConfig, loss::MseLoss, DropoutConfig, PaddingConfig2d, Sigmoid};

type fX = f32;

const W_SMALL: usize = 700;
const H_SMALL: usize = 700;
const W_LARGE: usize = 1400;
const H_LARGE: usize = 1400;
const INTERMEDIATE_FEATURES: usize = 16;

#[derive(Parser)]
#[command(version, about = "Image super-resolution trainer", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        train_small_dir: PathBuf,
        train_large_dir: PathBuf,
        valid_small_dir: PathBuf,
        valid_large_dir: PathBuf,
        output_dir: PathBuf,
        factor: usize,

        #[arg(long, default_value_t = 1e-4)]
        lr: fX,

        #[arg(long, default_value_t = 100)]
        epochs: usize,

        /// Moving average window for reporting
        #[arg(long, default_value_t = 100)]
        recent_window: usize,
    },
}

struct ImageSRDataset {
    small_dir: PathBuf,
    large_dir: PathBuf,
    count: usize,
}
impl ImageSRDataset {
    fn load(small_dir: &Path, large_dir: &Path) -> Self {
        let small_count = std::fs::read_dir(small_dir).unwrap().count();
        let large_count = std::fs::read_dir(large_dir).unwrap().count();

        assert_eq!(small_count, large_count);

        Self {
            small_dir: small_dir.to_path_buf(),
            large_dir: large_dir.to_path_buf(),
            count: small_count,
        }
    }
}

impl Dataset<ImageSRItem> for ImageSRDataset {
    fn get(&self, index: usize) -> Option<ImageSRItem> {
        let mut small_path = self.small_dir.clone();
        small_path.push(index.to_string());

        let mut large_path = self.large_dir.clone();
        large_path.push(index.to_string());

        Some(ImageSRItem {
            small_path,
            large_path,
        })
    }
    fn len(&self) -> usize {
        self.count
    }
}

#[derive(Debug, Clone)]
struct ImageSRItem {
    small_path: PathBuf,
    large_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ImageSRBatch<B: Backend> {
    pub small: Option<Tensor<B, 4>>,
    pub large: Option<Tensor<B, 4>>,
}

#[derive(Clone)]
struct ImageSRBatcher<B: Backend> {
    device: B::Device,
    small_min_width: usize,
    small_min_height: usize,
    large_min_width: usize,
    large_min_height: usize,

    /// Factor needs to be known so we can pre-upscale "small"
    pub factor: usize,
}

impl<B: Backend> ImageSRBatcher<B> {
    fn load_img(img_path: &Path) -> ImageResult<DynamicImage> {
        ImageReader::open(img_path)?.with_guessed_format()?.decode()
    }

    /// If the image is smaller than the specified minimums, Ok(None) is returned
    /// Otherwise, the image cropped to the minimums is returned
    fn load_cropped_img_as_tensor(
        &self,
        img_path: &Path,
        dev: &B::Device,
        min_width: usize,
        min_height: usize,
        factor: usize,
        normalize: bool,
    ) -> ImageResult<Option<Tensor<B, 3>>> {
        let img = Self::load_img(img_path)?;
        let w = img.width() as usize;
        let h = img.height() as usize;

        if w < min_width || h < min_height {
            // eprintln!(
            //     "Undersized: {} wxh: {}x{} mins: {}x{} factor: {} normalize: {}",
            //     img_path.display(),
            //     w,
            //     h,
            //     min_width,
            //     min_height,
            //     factor,
            //     normalize
            // );
            return Ok(None);
        }

        let new_w = w * factor;
        let new_h = h * factor;

        // Pre-upscale using a standard algorithm
        let img = resize(&img, new_w as u32, new_h as u32, FilterType::Nearest);

        let img = DynamicImage::ImageRgba8(img);

        let bytes = img.into_rgb32f().into_vec();

        let flat_tensor: Tensor<B, 1> = Tensor::from_floats(&bytes[..], dev);

        // We know this because of the into_rgb32f() call which forces it to RGB
        let c = 3usize;

        // The data layout appears to be (w, h, c), see ImageBuffer::pixel_indices_unchecked
        let img = flat_tensor.reshape([new_w, new_h, c]).slice([
            Some((0i64, (min_width * factor) as i64)),
            Some((0i64, (min_height * factor) as i64)),
            None,
        ]);

        // 0:width
        // 1:height
        // 2:channels
        let img = img.swap_dims(0, 2);

        let img = img * if normalize { 255.0 } else { 1.0 };

        Ok(Some(img))
    }
}

impl<B: Backend> Batcher<ImageSRItem, ImageSRBatch<B>> for ImageSRBatcher<B> {
    fn batch(&self, items: Vec<ImageSRItem>) -> ImageSRBatch<B> {
        let mut small: Vec<Tensor<B, 3>> = Vec::with_capacity(items.len());
        let mut large: Vec<Tensor<B, 3>> = Vec::with_capacity(items.len());

        let _small: Vec<Option<Tensor<B, 3>>> = items
            .iter()
            .map(|p| {
                let r = self.load_cropped_img_as_tensor(
                    &p.small_path,
                    &self.device,
                    self.small_min_width,
                    self.small_min_height,
                    self.factor,
                    true,
                );

                match r {
                    Err(e) => {
                        eprintln!("Image load error: {}", e);
                        None
                    }
                    Ok(t) => t,
                }
            })
            .collect();

        let _large: Vec<Option<Tensor<B, 3>>> = items
            .into_iter()
            .map(|p| {
                let r = self.load_cropped_img_as_tensor(
                    &p.large_path,
                    &self.device,
                    self.large_min_width,
                    self.large_min_height,
                    1,
                    false,
                );

                match r {
                    Err(e) => {
                        eprintln!("Image load error: {}", e);
                        None
                    }
                    Ok(t) => t,
                }
            })
            .collect();

        debug_assert_eq!(_small.len(), _large.len());

        for (small_img, large_img) in std::iter::zip(_small, _large) {
            if let Some(small_img) = small_img {
                if let Some(large_img) = large_img {
                    small.push(small_img);
                    large.push(large_img);
                } else {
                    eprintln!("small ok, large bad");
                }
            } else if large_img.is_some() {
                eprintln!("small bad, large ok");
            } else {
                // both bad, not surprising
            }
        }

        let small = if small.is_empty() {
            None
        } else {
            Some(Tensor::stack::<4>(small, 0))
        };
        let large = if large.is_empty() {
            None
        } else {
            Some(Tensor::stack::<4>(large, 0))
        };

        if let Some(small) = small.as_ref() {
            if let Some(large) = large.as_ref() {
                assert_eq!(small.shape().dims[0], large.shape().dims[0]);
                assert_eq!(small.shape().dims[1], large.shape().dims[1]);
            }
        }

        ImageSRBatch { small, large }
    }
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
    activation: Sigmoid,
}
impl<B: Backend> Model<B> {
    /// Refine an already-upscaled image to look more realistic and remove jpeg artifacts
    ///
    /// # Shapes
    /// - Small images (batch, width, height, channel)
    fn upscale(&self, small: Tensor<B, 4>, training: bool) -> Tensor<B, 4> {
        // the input is in [0, 1]

        let mut x = self.deep.forward(small);
        if training {
            x = self.dropout.forward(x);
        }
        let x = self.activation.forward(x);

        let mut x = self.deeper.forward(x);
        if training {
            x = self.dropout.forward(x);
        }
        let x = self.activation.forward(x);

        let mut x = self.deepest.forward(x);
        if training {
            x = self.dropout.forward(x);
        }

        // the output is in [0, 255]
        self.activation.forward(x) * 255.0
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.1")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            deep: Conv2dConfig::new([3, INTERMEDIATE_FEATURES], [7, 7])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            deeper: Conv2dConfig::new([INTERMEDIATE_FEATURES, INTERMEDIATE_FEATURES], [5, 5])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            deepest: Conv2dConfig::new([INTERMEDIATE_FEATURES, 3], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            // pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Sigmoid::new(),
            // linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            // linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Config)]
pub struct ImageSRTrainingConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 248949)]
    pub seed: u64,

    #[config(default = 1e-5)]
    pub lr: f64,

    pub model: ModelConfig,

    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(
    device: &B::Device,
    train_small_dir: &Path,
    train_large_dir: &Path,
    valid_small_dir: &Path,
    valid_large_dir: &Path,
    output_dir: &Path,
    factor: usize,
) {
    let config_model = ModelConfig { dropout: 0.2 };
    let config_optimizer = AdamConfig::new();
    let config = ImageSRTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    let mut model: Model<B> = config.model.init(device);
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let train_batcher: ImageSRBatcher<B> = ImageSRBatcher {
        device: device.clone(),
        small_min_width: W_SMALL,
        small_min_height: H_SMALL,
        large_min_width: W_LARGE,
        large_min_height: H_LARGE,
        factor,
    };
    let valid_batcher: ImageSRBatcher<B::InnerBackend> = ImageSRBatcher {
        device: device.clone(),
        small_min_width: W_SMALL,
        small_min_height: H_SMALL,
        large_min_width: W_LARGE,
        large_min_height: H_LARGE,
        factor,
    };

    let dataloader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageSRDataset::load(train_small_dir, train_large_dir));

    let dataloader_test = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageSRDataset::load(valid_small_dir, valid_large_dir));

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            if let Some(small) = batch.small {
                if let Some(large) = batch.large {
                    let pred = model.upscale(small, true);

                    assert_eq!(pred.shape(), large.shape());

                    let loss = MseLoss::new().forward(pred.clone(), large, Reduction::Mean);

                    print!(
                        "\r[Train - Epoch {} - Iteration {}] Loss {:.3}          ",
                        epoch,
                        iteration,
                        loss.clone().into_scalar(),
                    );
                    stdout().flush().unwrap();

                    // Gradients for the current backward pass
                    let grads = loss.backward();

                    // Gradients linked to each parameter of the model.
                    let grads = GradientsParams::from_grads(grads, &model);

                    // Update the model using the optimizer.
                    model = optim.step(config.lr, model, grads);
                } else {
                    eprintln!("large was None");
                }
            } else {
                eprintln!("small was None");
            }
        }

        println!();

        let model_path = {
            let mut p = output_dir.to_path_buf();
            p.push(epoch.to_string());
            p
        };
        model
            .clone()
            .save_file(model_path, &recorder)
            .expect("Should be able to save the model");

        let mut total_valid_loss = 0.0f64;
        let mut valid_batches = 0usize;
        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            if let Some(small) = batch.small {
                if let Some(large) = batch.large {
                    let pred = model_valid.upscale(small, false);
                    let loss = MseLoss::new().forward(pred.clone(), large, Reduction::Mean);
                    let loss = loss.into_scalar();

                    total_valid_loss += loss.to_f64();
                    valid_batches += 1;

                    print!(
                        "\r[Valid - Epoch {} - Iteration {}] Loss {}",
                        epoch, iteration, loss,
                    );
                    stdout().flush().unwrap();
                } else {
                    eprintln!("valid large was None");
                }
            } else {
                eprintln!("valid small was None");
            }
        }
        println!();

        let mean_valid_loss = total_valid_loss / valid_batches as f64;
        println!("Mean validation loss: {}", mean_valid_loss);
    }
}

fn main() -> Result<(), walkdir::Error> {
    let cli = Cli::parse();

    type B = Wgpu<f32, i32>;
    let dev = Default::default();

    match &cli.command {
        Commands::Train {
            train_small_dir,
            train_large_dir,
            valid_small_dir,
            valid_large_dir,
            output_dir,
            lr,
            epochs,
            recent_window,
            factor,
        } => {
            run::<Autodiff<B>>(
                &dev,
                train_small_dir,
                train_large_dir,
                valid_small_dir,
                valid_large_dir,
                output_dir,
                *factor,
            );
        }
    }

    Ok(())
}
