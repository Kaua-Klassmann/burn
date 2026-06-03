use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    module::AutodiffModule,
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear,
        LinearConfig, loss::BinaryCrossEntropyLossConfig,
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::activation::sigmoid,
};

fn read_dataset() -> (Vec<f32>, Vec<i32>) {
    let text = std::fs::read_to_string("datasets/wdbc.data").unwrap();
    let lines = text.split("\n").collect::<Vec<&str>>();

    let mut features = Vec::with_capacity(lines.len() * 30);
    let mut targets = Vec::with_capacity(lines.len());

    for line in lines {
        let values = line.split(",").collect::<Vec<&str>>();

        features.extend(values[2..].iter().map(|v| v.parse::<f32>().unwrap()));
        targets.push(if values[1] == "M" { 1 } else { 0 });
    }

    (features, targets)
}

type B = Autodiff<Wgpu>;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    l1: Linear<B>,
    bat1: BatchNorm<B>,
    act1: LeakyRelu,
    do1: Dropout,
    l2: Linear<B>,
    bat2: BatchNorm<B>,
    act2: LeakyRelu,
    do2: Dropout,
    l3: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(x);
        let x = self.bat1.forward(x);
        let x = self.act1.forward(x);
        let x = self.do1.forward(x);
        let x = self.l2.forward(x);
        let x = self.bat2.forward(x);
        let x = self.act2.forward(x);
        let x = self.do2.forward(x);
        self.l3.forward(x)
    }
}

fn main() {
    let device = &WgpuDevice::default();

    // prepara dataset
    let (f, t) = read_dataset();

    let mut features =
        Tensor::<B, 1, Float>::from_floats(f.as_slice(), device).reshape([t.len(), 30]);
    let targets = Tensor::<B, 1, Int>::from_ints(t.as_slice(), device).reshape([t.len(), 1]);

    drop(f);
    drop(t);

    // normaliza os dados
    let mean = features.clone().mean_dim(0);
    let std = features.clone().var(0).sqrt();

    features = (features - mean) / (std);

    let train_lines = targets.dims()[0] * 8 / 10;

    let x_train = features.clone().slice([0..train_lines]);
    let x_test = features.slice([train_lines..]);

    let y_train = targets.clone().slice([0..train_lines]);
    let y_test = targets.slice([train_lines..]);

    // modelagem da rede
    let mut model = Model {
        l1: LinearConfig::new(30, 32).init(device),
        bat1: BatchNormConfig::new(32).init(device),
        act1: LeakyReluConfig::new().with_negative_slope(0.01).init(),
        do1: DropoutConfig::new(0.2).init(),
        l2: LinearConfig::new(32, 16).init(device),
        bat2: BatchNormConfig::new(16).init(device),
        act2: LeakyReluConfig::new().with_negative_slope(0.01).init(),
        do2: DropoutConfig::new(0.2).init(),
        l3: LinearConfig::new(16, 1).init(device),
    };
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();

    if Path::new("models/binary.mpk").exists() {
        model = model
            .load_file(
                "models/binary",
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                device,
            )
            .unwrap();
    };

    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(device);

    for epoch in 0..=1000 {
        let pred = model.forward(x_train.clone());

        let loss = loss_fn.forward(pred.clone(), y_train.clone());

        let grads = GradientsParams::from_grads(loss.backward(), &model);

        model = opt.step(1e-3, model, grads);

        if epoch % 100 == 0 {
            let pred = model.valid().forward(x_test.clone().inner());

            let loss_test = loss_fn
                .valid()
                .forward(pred.clone(), y_test.clone().inner());

            let acc: f32 = sigmoid(pred)
                .greater_elem(0.5)
                .equal(y_test.clone().inner().equal_elem(1))
                .float()
                .mean()
                .into_scalar()
                * 100.;

            println!(
                "Epoch {} - Loss treino {} - Loss test {} - Acc {}",
                epoch,
                loss.into_scalar(),
                loss_test.into_scalar(),
                acc
            );
        }
    }

    model
        .save_file(
            "models/binary",
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap()
}
