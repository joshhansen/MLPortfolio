use std::{
    io::{stdout, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    backend::{
        wgpu::{JitBackend, WgpuDevice, WgpuRuntime},
        Autodiff, Wgpu,
    },
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
use rand::Rng;

type fX = f32;

const SMALL_MIN_W: usize = 70;
const SMALL_MIN_H: usize = 70;

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

        #[arg(long, short = 'd', default_value = "0", value_parser = parse_device)]
        dev: Vec<WgpuDevice>,

        #[arg(long, short = 'S', default_value_t = 10)]
        samples_per_img: usize,
    },
}

fn parse_device(s: &str) -> Result<WgpuDevice, String> {
    if s == "cpu" {
        return Ok(WgpuDevice::Cpu);
    }

    let idx: usize = s
        .parse()
        .map_err(|_| format!("Unrecognized device: {}", s))?;

    Ok(WgpuDevice::DiscreteGpu(idx))
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

    /// Factor needs to be known so we can pre-upscale "small"
    pub factor: usize,

    /// The number of sub-images to extract form each image
    samples: usize,
}

impl<B: Backend> ImageSRBatcher<B> {
    fn load_img(img_path: &Path) -> ImageResult<DynamicImage> {
        ImageReader::open(img_path)?.with_guessed_format()?.decode()
    }

    /// If an image is smaller than the specified minimums, Ok(None) is returned
    /// Otherwise, `samples` random sub-images are returned in stacked tensors,
    /// the first for 'small', the second for 'large'
    fn load_cropped_imgs_as_tensors(
        small_img_path: &Path,
        large_img_path: &Path,
        dev: &B::Device,
        small_min_width: usize,
        small_min_height: usize,
        factor: usize,
        samples: usize,
    ) -> ImageResult<Option<(Vec<Tensor<B, 3>>, Vec<Tensor<B, 3>>)>> {
        let small = Self::load_img(small_img_path)?;
        let large = Self::load_img(large_img_path)?;

        let small_w = small.width() as usize;
        let small_h = small.height() as usize;

        if small_w < small_min_width || small_h < small_min_height {
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

        let large_min_width = small_min_width * factor;
        let large_min_height = small_min_height * factor;

        let large_w = large.width() as usize;
        let large_h = large.height() as usize;

        if large_w < large_min_width || large_h < large_min_height {
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

        let mut small_imgs: Vec<Tensor<B, 3>> = Vec::with_capacity(samples);
        let mut large_imgs: Vec<Tensor<B, 3>> = Vec::with_capacity(samples);

        let small_max_x = small.width() as usize - small_min_width;
        let small_max_y = small.height() as usize - small_min_height;

        let mut rng = rand::thread_rng();

        for _ in 0..samples {
            let small_x = rng.gen_range(0..small_max_x) as u32;
            let small_y = rng.gen_range(0..small_max_y) as u32;

            let large_x = small_x * factor as u32;
            let large_y = small_y * factor as u32;

            let small_img = small.crop_imm(
                small_x,
                small_y,
                small_min_width as u32,
                small_min_height as u32,
            );
            let large_img = large.crop_imm(
                large_x,
                large_y,
                large_min_width as u32,
                large_min_height as u32,
            );

            let small_img = resize(
                &small_img,
                large_min_width as u32,
                large_min_height as u32,
                FilterType::Nearest,
            );

            let small_img = DynamicImage::ImageRgba8(small_img);

            let small_img = Self::img_to_tensor(small_img, true, dev); //FIXME
            let large_img = Self::img_to_tensor(large_img, false, dev); //FIXME

            small_imgs.push(small_img);
            large_imgs.push(large_img);
        }

        Ok(Some((small_imgs, large_imgs)))
    }

    fn img_to_tensor(img: DynamicImage, normalize: bool, dev: &B::Device) -> Tensor<B, 3> {
        let w = img.width() as usize;
        let h = img.height() as usize;

        let bytes = img.into_rgb32f().into_vec();

        let flat_tensor: Tensor<B, 1> = Tensor::from_floats(&bytes[..], dev);
        println!("Flat tensor shape: {:?}", flat_tensor.shape());

        // We know this because of the into_rgb32f() call which forces it to RGB
        let c = 3usize;

        // The data layout appears to be (w, h, c), see ImageBuffer::pixel_indices_unchecked
        let img = flat_tensor.reshape([w, h, c]);

        // 0:width
        // 1:height
        // 2:channels
        let img = img.swap_dims(0, 2);

        let img = img * if normalize { 255.0 } else { 1.0 };

        // Set the root of autodiff here
        img.detach()
    }

    // /// If the image is smaller than the specified minimums, Ok(None) is returned
    // /// Otherwise, `samples` random sub-images are returned in a stacked tensor
    // fn load_cropped_imgs_as_tensor_old(
    //     img_path: &Path,
    //     dev: &B::Device,
    //     min_width: usize,
    //     min_height: usize,
    //     factor: usize,
    //     sample_locs: Option<Vec<(usize, usize)>>,
    //     normalize: bool,
    // ) -> ImageResult<Option<Tensor<B, 4>>> {
    //     // let img: Tensor<B, 3> = Tensor::random(
    //     //     vec![3, min_width * factor, min_height * factor],
    //     //     Distribution::Normal(0f64, 1f64),
    //     //     dev,
    //     // );

    //     let img = Self::load_img(img_path)?;
    //     let w = img.width() as usize;
    //     let h = img.height() as usize;

    //     if w < min_width || h < min_height {
    //         // eprintln!(
    //         //     "Undersized: {} wxh: {}x{} mins: {}x{} factor: {} normalize: {}",
    //         //     img_path.display(),
    //         //     w,
    //         //     h,
    //         //     min_width,
    //         //     min_height,
    //         //     factor,
    //         //     normalize
    //         // );
    //         return Ok(None);
    //     }

    //     let mut imgs: Vec<Tensor<B, 3>> = Vec::with_capacity(sample_locs.len());

    //     for (x, y) in sample_locs {
    //         let x = x as u32;
    //         let y = y as u32;
    //         let img = img.crop_imm(x, y, x + min_width as u32, y + min_height as u32);

    //         let new_w = min_width * factor;
    //         let new_h = min_height * factor;

    //         // Pre-upscale using a standard algorithm
    //         let img = resize(&img, new_w as u32, new_h as u32, FilterType::Nearest);

    //         let img = DynamicImage::ImageRgba8(img);

    //         let bytes = img.into_rgb32f().into_vec();

    //         let flat_tensor: Tensor<B, 1> = Tensor::from_floats(&bytes[..], dev);

    //         // We know this because of the into_rgb32f() call which forces it to RGB
    //         let c = 3usize;

    //         // The data layout appears to be (w, h, c), see ImageBuffer::pixel_indices_unchecked
    //         let img = flat_tensor.reshape([new_w, new_h, c]).slice([
    //             Some((0i64, (min_width * factor) as i64)),
    //             Some((0i64, (min_height * factor) as i64)),
    //             None,
    //         ]);

    //         // 0:width
    //         // 1:height
    //         // 2:channels
    //         let img = img.swap_dims(0, 2);

    //         let img = img * if normalize { 255.0 } else { 1.0 };

    //         // Set the root of autodiff here
    //         let img = img.detach();

    //         imgs.push(img);
    //     }

    //     let imgs: Tensor<B, 4> = Tensor::stack(imgs, 0);

    //     println!("Loaded {}", img_path.display());

    //     Ok(Some(imgs))
    // }
}

impl<B: Backend> Batcher<ImageSRItem, ImageSRBatch<B>> for ImageSRBatcher<B> {
    fn batch(&self, items: Vec<ImageSRItem>) -> ImageSRBatch<B> {
        println!("Building batch of {}", items.len());
        let mut small: Vec<Tensor<B, 3>> = Vec::with_capacity(items.len());
        let mut large: Vec<Tensor<B, 3>> = Vec::with_capacity(items.len());

        let both: Vec<Option<(Vec<Tensor<B, 3>>, Vec<Tensor<B, 3>>)>> = items
            .into_iter()
            .map(|p| {
                let r = Self::load_cropped_imgs_as_tensors(
                    &p.small_path,
                    &p.large_path,
                    &self.device,
                    self.small_min_width,
                    self.small_min_height,
                    self.factor,
                    self.samples,
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

        for both_from_img in both {
            if let Some(both_from_img) = both_from_img {
                let (small_imgs, large_imgs) = both_from_img;
                small.extend(small_imgs);
                large.extend(large_imgs);
            } else {
                // something was mis-sized, do nothing about it here
            }
        }
        // for (small_img, large_img) in std::iter::zip(_small, _large) {
        //     if let Some(small_img) = small_img {
        //         if let Some(large_img) = large_img {
        //             small.push(small_img);
        //             large.push(large_img);
        //         } else {
        //             eprintln!("small ok, large bad");
        //         }
        //     } else if large_img.is_some() {
        //         eprintln!("small bad, large ok");
        //     } else {
        //         // both bad, not surprising
        //     }
        // }

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
            println!("small shape: {:?}", small.shape());

            if let Some(large) = large.as_ref() {
                println!("large shape: {:?}", large.shape());
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

        println!("Built");

        ImageSRBatch { small, large }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    deep: Conv2d<B>,
    deeper: Conv2d<B>,
    deepest: Conv2d<B>,
    dropout: Dropout,
    relu: Relu,
    sigmoid: Sigmoid,
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
    fn upscale(&self, small: Tensor<B, 4>, training: bool) -> Tensor<B, 4> {
        // the input is in [0, 1]

        // println!("Are nans? {}", small.is_nan().any());

        let mut x = self.deep.forward(small.clone());
        if training {
            x = self.dropout.forward(x);
        }
        let x = self.relu.forward(x);

        let mut x = self.deeper.forward(x);
        if training {
            x = self.dropout.forward(x);
        }
        let x = self.relu.forward(x);

        let mut x = self.deepest.forward(x);
        if training {
            x = self.dropout.forward(x);
        }

        assert_eq!(small.shape(), x.shape());

        // The skip/residual connection
        let x = small + x;

        // the output is in [0, 255]
        self.sigmoid.forward(x) * 255.0

        // small
        // let x = self.deep.forward(small);

        // self.deep.forward(small)
        // self.deeper.forward(self.deep.forward(small))
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
            deep: Conv2dConfig::new([3, INTERMEDIATE_FEATURES], [1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            deeper: Conv2dConfig::new([3, INTERMEDIATE_FEATURES], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            deepest: Conv2dConfig::new([INTERMEDIATE_FEATURES, 3], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            relu: Relu::new(),
            sigmoid: Sigmoid::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Config)]
pub struct ImageSRTrainingConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 10)]
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
    samples_per_img: usize,
) {
    println!("Running");
    let config_model = ModelConfig { dropout: 0.2 };
    let config_optimizer = AdamConfig::new();
    let config = ImageSRTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    let mut model: Model<B> = config.model.init(device);
    let mut optim = config.optimizer.init::<B, Model<B>>();

    println!("Optim");

    let train_batcher: ImageSRBatcher<B> = ImageSRBatcher {
        device: device.clone(),
        small_min_width: SMALL_MIN_W,
        small_min_height: SMALL_MIN_H,
        factor,
        samples: samples_per_img,
    };
    let valid_batcher: ImageSRBatcher<B::InnerBackend> = ImageSRBatcher {
        device: device.clone(),
        small_min_width: SMALL_MIN_W,
        small_min_height: SMALL_MIN_H,
        factor,
        samples: samples_per_img,
    };

    println!("Batchers");

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

    println!("Dataloaders");
    stdout().flush().unwrap();

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    println!("Recorder");

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            if let Some(small) = batch.small {
                if let Some(large) = batch.large {
                    println!("Going to upscale");
                    let pred = model.upscale(small, true);

                    assert_eq!(pred.shape(), large.shape());

                    println!("Going to loss");

                    let loss = MseLoss::new().forward(pred.clone(), large, Reduction::Mean);

                    println!(
                        "\r[Train - Epoch {} - Iteration {}] Loss {:.3}          ",
                        epoch,
                        iteration,
                        loss.clone().into_scalar(),
                    );
                    stdout().flush().unwrap();

                    // Gradients for the current backward pass
                    let grads = loss.backward();
                    println!("Backward");
                    stdout().flush().unwrap();

                    // Gradients linked to each parameter of the model.
                    let grads = GradientsParams::from_grads(grads, &model);
                    println!("Grads");
                    stdout().flush().unwrap();

                    // Update the model using the optimizer.
                    model = optim.step(config.lr, model, grads);
                    println!("Optim step");
                    stdout().flush().unwrap();
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
struct ModelTensors<B: Backend> {
    deep: Conv2dTensors<B>,
    // deeper: Conv2dTensors<B>,
    // deepest: Conv2dTensors<B>,
}
impl<B: Backend> ModelTensors<B> {
    fn to_device(self, dev: &B::Device) -> Self {
        Self {
            deep: self.deep.to_device(dev),
            // deeper: self.deeper.to_device(dev),
            // deepest: self.deepest.to_device(dev),
        }
    }
}
impl<B: Backend> From<Model<B>> for ModelTensors<B> {
    fn from(m: Model<B>) -> Self {
        Self {
            deep: Conv2dTensors::from(m.deep),
            // deeper: Conv2dTensors::from(m.deeper),
            // deepest: Conv2dTensors::from(m.deepest),
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
    samples_per_img: usize,
) {
    type B = Autodiff<Wgpu>;
    let config_model = ModelConfig { dropout: 0.2 };
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
                small_min_width: SMALL_MIN_W,
                small_min_height: SMALL_MIN_H,
                factor,
                samples: samples_per_img,
            };
            let valid_batcher: ImageSRBatcher<<B as AutodiffBackend>::InnerBackend> =
                ImageSRBatcher {
                    device: device.clone(),
                    small_min_width: SMALL_MIN_W,
                    small_min_height: SMALL_MIN_H,
                    factor,
                    samples: samples_per_img,
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

                            println!(
                                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                                epoch,
                                iteration,
                                loss.clone().into_scalar(),
                                // accuracy,
                            );

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

                            println!(
                                "[Valid - Epoch {} - Iteration {}] Loss {}",
                                epoch,
                                iteration,
                                loss.clone().into_scalar(),
                            );
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
        mean_model.deep = mean_conv2d_from_parts(
            mean_model.deep,
            models
                .iter()
                .map(|m| m.as_ref().unwrap().deep.clone())
                .collect(),
            &control_dev,
        );
        // mean_model.deeper = mean_conv2d_from_parts(
        //     mean_model.deeper,
        //     models
        //         .iter()
        //         .map(|m| m.as_ref().unwrap().deeper.clone())
        //         .collect(),
        //     &control_dev,
        // );
        // mean_model.deepest = mean_conv2d_from_parts(
        //     mean_model.deepest,
        //     models
        //         .iter()
        //         .map(|m| m.as_ref().unwrap().deepest.clone())
        //         .collect(),
        //     &control_dev,
        // );
    }
}

fn main() -> Result<(), walkdir::Error> {
    let cli = Cli::parse();

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
            samples_per_img,
            dev,
        } => {
            let d = &dev[0];
            type B = JitBackend<WgpuRuntime, f32, i32>;

            // burn::backend::wgpu::init_sync::<burn::backend::wgpu::OpenGl>(d, Default::default());

            // type B = Wgpu<f32, i32>;

            if dev.len() > 1 {
                eprintln!("Only using the first device ({:?})", d);
            }

            run::<Autodiff<B>>(
                d,
                train_small_dir,
                train_large_dir,
                valid_small_dir,
                valid_large_dir,
                output_dir,
                *factor,
                *samples_per_img,
            );
        }
    }

    Ok(())
}
