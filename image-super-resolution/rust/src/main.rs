use std::{
    collections::VecDeque,
    io::{stdout, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    module::{AutodiffModule, Param},
    nn::{conv::Conv2d, loss::Reduction, Dropout, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
    tensor::{backend::AutodiffBackend, cast::ToElement, BasicOps, Numeric, TensorKind},
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

        #[arg(long, short, default_value_t = 3)]
        units: usize,
    },
}

struct ImageSRDataset {
    small_dir: PathBuf,
    large_dir: PathBuf,
    count: usize,
    // number of partitions to break the dataset into, in order to shard loading
    // set to 1 if not partitioning
    partitions: usize,

    // the partition to select
    // 0-indexed
    partition: usize,
}
impl ImageSRDataset {
    fn load(small_dir: &Path, large_dir: &Path, partitions: usize, partition: usize) -> Self {
        let small_count = std::fs::read_dir(small_dir).unwrap().count();
        let large_count = std::fs::read_dir(large_dir).unwrap().count();

        assert_eq!(small_count, large_count);

        Self {
            small_dir: small_dir.to_path_buf(),
            large_dir: large_dir.to_path_buf(),
            count: small_count,
            partitions,
            partition,
        }
    }
}

impl Dataset<ImageSRItem> for ImageSRDataset {
    fn get(&self, index: usize) -> Option<ImageSRItem> {
        let p = index % self.partitions;
        if p != self.partition {
            return None;
        }

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

        // Set the root of autodiff here
        let img = img.detach();

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

        assert_eq!(
            small.as_ref().map(|s| s.shape().dims[0]),
            large.as_ref().map(|l| l.shape().dims[0])
        );
        assert_eq!(
            small.as_ref().map(|s| s.shape().dims[1]),
            large.as_ref().map(|l| l.shape().dims[1])
        );

        ImageSRBatch { small, large }
    }
}

#[derive(Module, Debug)]
pub struct ResUnit<B: Backend> {
    convs: Vec<Conv2d<B>>,
    dropout: Dropout,
    relu: Relu,
    sigmoid: Sigmoid,
}
impl<B: Backend> ResUnit<B> {
    /// Input is in [0, 1]
    /// Output is in [0, 1]
    fn forward(&self, mut x: Tensor<B, 4>, train: bool) -> Tensor<B, 4> {
        let skip = x.clone();

        for conv in self.convs.iter() {
            x = conv.forward(x);
            if train {
                x = self.dropout.forward(x);
            }
            x = self.relu.forward(x);
        }

        assert_eq!(skip.shape(), x.shape());

        x = skip + x;

        self.sigmoid.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ResUnitConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

impl ResUnitConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResUnit<B> {
        ResUnit {
            convs: vec![
                Conv2dConfig::new([3, INTERMEDIATE_FEATURES], [7, 7])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                Conv2dConfig::new([INTERMEDIATE_FEATURES, INTERMEDIATE_FEATURES], [5, 5])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                Conv2dConfig::new([INTERMEDIATE_FEATURES, 3], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            ],
            relu: Relu::new(),
            sigmoid: Sigmoid::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    units: Vec<ResUnit<B>>,
}
impl<B: Backend> Model<B> {
    /// Refine an already-upscaled image to look more realistic and remove jpeg artifacts
    ///
    /// A single resnet with three conv layers
    ///
    /// The "small" input image should actually be the intended size already
    /// This model only refines an already-upscaled image.
    /// Use a nearest-neighbor upscale first.
    ///
    /// # Shapes
    /// - Small images (batch, width, height, channel)
    fn upscale(&self, mut x: Tensor<B, 4>, train: bool) -> Tensor<B, 4> {
        // the input is in [0, 1]

        for unit in &self.units {
            x = unit.forward(x, train);
        }

        // output is in [0, 255]
        x * 255.0
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 3)]
    pub units: usize,
    unit_config: ResUnitConfig,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            units: (0..self.units)
                .map(|_| self.unit_config.init(device))
                .collect(),
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
    units: usize,
) {
    let config_unit = ResUnitConfig::new();
    let config_model = {
        let mut c = ModelConfig::new(config_unit);
        c.units = units;
        c
    };
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
        .build(ImageSRDataset::load(train_small_dir, train_large_dir, 1, 0));

    let dataloader_test = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageSRDataset::load(valid_small_dir, valid_large_dir, 1, 0));

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

fn mean_tensor<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B> + Numeric<B>>(
    tensors: Vec<Tensor<B, D, K>>,
    dev: &B::Device,
) -> Tensor<B, D, K> {
    let mut sum = Tensor::zeros(tensors[0].shape(), dev);

    let l = tensors.len() as u64;

    for t in tensors {
        sum = sum + t;
    }

    sum / l
}

/// Returns the first conv, with updated weights and bias, located on `dev`
fn mean_conv2d<B: Backend>(convs: Vec<Conv2d<B>>, dev: &B::Device) -> Conv2d<B> {
    let mut c = convs[0].clone().to_device(dev);
    let bias_tensors: Vec<Tensor<B, 1>> = convs
        .iter()
        .map(|c| c.bias.clone().unwrap().into_value())
        .collect();
    let weight_tensors: Vec<Tensor<B, 4>> =
        convs.into_iter().map(|c| c.weight.into_value()).collect();
    c.weight = Param::from_tensor(mean_tensor(weight_tensors, dev));
    c.bias = Some(Param::from_tensor(mean_tensor(bias_tensors, dev)));

    c
}

fn mean_conv2d_from_parts<B: Backend>(
    mut c: Conv2d<B>,
    parts: Vec<Conv2dTensors<B>>,
    dev: &B::Device,
) -> Conv2d<B> {
    let mut weight = Vec::with_capacity(parts.len());
    let mut bias = Vec::with_capacity(parts.len());

    for p in parts {
        weight.push(p.weight.into());
        bias.push(p.bias.unwrap().into());
    }

    c.weight = Param::from_tensor(mean_tensor(weight, dev));
    c.bias = Some(Param::from_tensor(mean_tensor(bias, dev)));

    c
}

fn mean_resunit_from_parts<B: Backend>(
    mut u: ResUnit<B>,
    mut parts: Vec<ResUnitTensors<B>>,
    dev: &B::Device,
) -> ResUnit<B> {
    let l = parts[0].convs.len();
    for p in &parts {
        assert_eq!(l, p.convs.len());
    }

    let mut src_convs: VecDeque<Conv2d<B>> = u.convs.into_iter().collect();

    let mut convs: Vec<Conv2d<B>> = Vec::with_capacity(l);

    let num_parts = parts.len();

    for _ in 0..l {
        let mut unit_parts: Vec<Conv2dTensors<B>> = Vec::with_capacity(num_parts);
        for p in &mut parts {
            unit_parts.push(p.convs.pop_front().unwrap());
        }

        let c = src_convs.pop_front().unwrap();

        convs.push(mean_conv2d_from_parts(c, unit_parts, dev));
    }

    u.convs = convs;

    u
}

#[derive(Clone)]
struct Conv2dTensors<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
}
impl<B: Backend> Conv2dTensors<B> {
    fn to_device(self, dev: &B::Device) -> Self {
        Self {
            weight: self.weight.to_device(dev),
            bias: self.bias.map(|b| b.to_device(dev)),
        }
    }
}
impl<B: Backend> From<Conv2d<B>> for Conv2dTensors<B> {
    fn from(m: Conv2d<B>) -> Self {
        Self {
            weight: m.weight.into_value(),
            bias: m.bias.map(|b| b.into_value()),
        }
    }
}

#[derive(Clone)]
struct ResUnitTensors<B: Backend> {
    convs: VecDeque<Conv2dTensors<B>>,
}
impl<B: Backend> ResUnitTensors<B> {
    fn to_device(self, dev: &B::Device) -> Self {
        Self {
            convs: self.convs.into_iter().map(|c| c.to_device(dev)).collect(),
        }
    }
}
impl<B: Backend> From<ResUnit<B>> for ResUnitTensors<B> {
    fn from(u: ResUnit<B>) -> Self {
        Self {
            convs: u.convs.into_iter().map(Conv2dTensors::from).collect(),
        }
    }
}

#[derive(Clone)]
struct ModelTensors<B: Backend> {
    units: Vec<ResUnitTensors<B>>,
}
impl<B: Backend> ModelTensors<B> {
    fn to_device(self, dev: &B::Device) -> Self {
        Self {
            units: self.units.into_iter().map(|u| u.to_device(dev)).collect(),
        }
    }
}
impl<B: Backend> From<Model<B>> for ModelTensors<B> {
    fn from(m: Model<B>) -> Self {
        Self {
            units: m.units.into_iter().map(ResUnitTensors::from).collect(),
        }
    }
}

use std::sync::mpsc::channel;
use std::sync::RwLock;
use std::thread;
fn run_multi(
    devices: Vec<WgpuDevice>,
    train_small_dir: &Path,
    train_large_dir: &Path,
    valid_small_dir: &Path,
    valid_large_dir: &Path,
    factor: usize,
) {
    type B = Autodiff<Wgpu>;
    let config_unit = ResUnitConfig::new();
    let config_model = ModelConfig::new(config_unit);

    let config_optimizer = AdamConfig::new();
    let config = ImageSRTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    let control_dev = WgpuDevice::Cpu;
    let shared_deep: Arc<RwLock<ModelTensors<B>>> = Arc::new(RwLock::new(ModelTensors::from(
        config.model.init(&control_dev),
    )));
    let (tx, rx) = channel();
    for gpu in 0..devices.len() {
        let tx = tx.clone();
        let devices = devices.clone();
        let config = config.clone();
        let train_small_dir = train_small_dir.to_path_buf();
        let train_large_dir = train_large_dir.to_path_buf();
        let valid_small_dir = valid_small_dir.to_path_buf();
        let valid_large_dir = valid_large_dir.to_path_buf();

        let device = devices[gpu].clone();
        let mut optim = config.optimizer.init::<B, Model<B>>();
        let mut model: Model<B> = config.model.init(&device);
        thread::spawn(move || loop {
            let train_batcher: ImageSRBatcher<B> = ImageSRBatcher {
                device: device.clone(),
                small_min_width: W_SMALL,
                small_min_height: H_SMALL,
                large_min_width: W_LARGE,
                large_min_height: H_LARGE,
                factor,
            };
            let valid_batcher: ImageSRBatcher<<B as AutodiffBackend>::InnerBackend> =
                ImageSRBatcher {
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
                .build(ImageSRDataset::load(
                    &train_small_dir,
                    &train_large_dir,
                    devices.len(),
                    gpu,
                ));

            let dataloader_test = DataLoaderBuilder::new(valid_batcher)
                .batch_size(config.batch_size)
                .shuffle(config.seed)
                .num_workers(config.num_workers)
                .build(ImageSRDataset::load(
                    &valid_small_dir,
                    &valid_large_dir,
                    devices.len(),
                    gpu,
                ));

            // Iterate over our training and validation loop for X epochs.
            for epoch in 1..config.num_epochs + 1 {
                // Implement our training loop.
                for (iteration, batch) in dataloader_train.iter().enumerate() {
                    if let Some(small) = batch.small {
                        if let Some(large) = batch.large {
                            let pred = model.upscale(small, true);
                            let loss = MseLoss::new().forward(pred, large, Reduction::Mean);

                            print!(
                                "\r[Train - Epoch {} - Iteration {}] Loss {:.3}",
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

                            tx.send((gpu, ModelTensors::from(model.clone()))).unwrap();
                        }
                    }
                }

                // Get the model without autodiff.
                let model_valid = model.valid();

                // Implement our validation loop.
                for (iteration, batch) in dataloader_test.iter().enumerate() {
                    if let Some(small) = batch.small {
                        if let Some(large) = batch.large {
                            let pred = model_valid.upscale(small, false);
                            let loss = MseLoss::new().forward(pred.clone(), large, Reduction::Mean);

                            print!(
                                "\r[Valid - Epoch {} - Iteration {}] Loss {}",
                                epoch,
                                iteration,
                                loss.clone().into_scalar(),
                            );
                            stdout().flush().unwrap();
                        }
                    }
                }
            }
        });
    }

    let mut models = vec![None; devices.len()];
    loop {
        let (gpu, gpu_model) = rx.recv().unwrap();
        models[gpu] = Some(gpu_model.to_device(&control_dev));

        let mut mean_model: Model<B> = config.model.init(&control_dev);

        mean_model.units = mean_model
            .units
            .into_iter()
            .enumerate()
            .map(|(unit_idx, u)| {
                let mut model_units: Vec<ResUnitTensors<B>> = Vec::with_capacity(models.len());
                for model_idx in 0..models.len() {
                    model_units.push(models[model_idx].as_ref().unwrap().units[unit_idx].clone());
                }
                mean_resunit_from_parts(u, model_units, &control_dev)
            })
            .collect();
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
            units,
        } => {
            run::<Autodiff<B>>(
                &dev,
                train_small_dir,
                train_large_dir,
                valid_small_dir,
                valid_large_dir,
                output_dir,
                *factor,
                *units,
            );
        }
    }

    Ok(())
}
