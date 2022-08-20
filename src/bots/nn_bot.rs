use super::*;

use std::collections::HashMap;
use std::io::{Read, Write, Cursor};
use std::fs::File;

use anyhow::{Result, Context};

use tch::{
    nn::{
        self,
        Conv2D,
        ConvConfig,
        Adam,
        Optimizer,
        OptimizerConfig,
        VarStore,
    },
    data::Iter2,
    Tensor,
    Kind,
    Device,
    Reduction,
    IndexOp,
};

#[derive(Clone, Debug)]
pub struct FeatureGen {
    state: State,
    seen_generals: Vec<isize>,
    seen_cities: Vec<bool>,
    seen_terrain: Vec<isize>,
}

impl FeatureGen {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            seen_generals: Vec::new(),
            seen_cities: Vec::new(),
            seen_terrain: Vec::new(),
        }
    }

    pub fn update(&mut self, state: &State) {
        self.seen_generals.resize(state.generals.len(), -1);
        self.seen_terrain .resize(state.terrain .len(), -1);
        self.seen_cities  .resize(state.terrain .len(), false);

        for i in 0..state.generals.len() {
            if state.generals[i] != -1 {
                self.seen_generals[i] = state.generals[i];
            }
        }

        for city in &state.cities {
            self.seen_cities[*city as usize] = true;
        }

        for i in 0..state.terrain.len() {
            if state.terrain[i] >= TILE_MOUNTAIN {
                self.seen_terrain[i] = state.terrain[i];
            }
        }

        self.state = state.clone();
    }
}

macro_rules! tensor_of_iter {
    ($x:expr) => {
        Tensor::of_slice(&$x.map(|x| x as i64 as f32).collect::<Vec<_>>())
    }
}

impl FeatureGen {
    pub fn generate_features(&self, player: usize, dev: Device) -> Tensor {
        let mut out = Vec::new();
        let size = self.seen_terrain.len();

        let my_max_armies = self.state.armies.iter()
            .enumerate()
            .filter(|(i, _)| self.state.terrain[*i] == player as isize)
            .map(|(_, x)| *x as i64)
            .max().unwrap();

        // my tiles
        out.push(tensor_of_iter!(self.seen_terrain.iter().map(|x| *x == player as isize)));
        // opponent tiles
        out.push(tensor_of_iter!(self.seen_terrain.iter().map(|x| *x != player as isize)));
        // my general
        out.push(tensor_of_iter!((0..size).map(|i| i as isize == self.seen_generals[player])));
        // opponent general
        out.push(tensor_of_iter!((0..size).map(|i| i as isize == self.seen_generals[player ^ 1])));
        // currently visible tiles
        out.push(tensor_of_iter!(self.state.terrain.iter().map(|x| *x >= TILE_MOUNTAIN)));
        // known cities
        out.push(tensor_of_iter!(self.seen_cities.iter().copied()));
        // known mountains
        out.push(tensor_of_iter!(self.seen_terrain.iter().map(|x| *x == TILE_MOUNTAIN)));
        // mountain/city in fog
        out.push(tensor_of_iter!(self.seen_terrain.iter().map(|x| *x == TILE_FOG_OBSTACLE)));
        // empty tile
        out.push(tensor_of_iter!(self.seen_terrain.iter().map(|x| *x == TILE_EMPTY || *x == TILE_FOG)));
        // armies (absolute)
        out.push(tensor_of_iter!(self.state.armies.iter().copied()));
        // armies (relative)
        out.push(tensor_of_iter!(self.state.armies.iter().copied()) / my_max_armies);

        Tensor::stack(&out, 0)
            .reshape(&[out.len() as i64, self.state.height as i64, self.state.width as i64])
            .to_device(dev)
    }
}

pub fn move_of_tensor(tensor: &Tensor, width: usize, height: usize) -> Move {
    let max = tensor
        .max_dim(0, false).1
        .int64_value(&[]) as usize;

    let start = max % (width * height);

    let end =
        match max / 8 % 4 {
            0 => start.saturating_sub(width),
            1 => start + 1,
            2 => start + width,
            _ => start.saturating_sub(1),
        };

    Move { start, end, is50: max / 8 >= 4 }
}

pub fn tensor_of_move(mov: Move, width: usize, height: usize) -> Tensor {
    let dir =
        match mov.end as i64 - mov.start as i64 {
            1   => 1,
            -1  => 3,
            0.. => 2,
            _   => 0,
        } + mov.is50 as usize * 4;

    let index = vec![Some(Tensor::of_slice(&[(mov.start + dir * width * height) as i64]))];

    // Tensor::of_slice(&[(mov.start + dir * width * height) as i32])

    Tensor::zeros(&[(8 * height * width) as i64], (Kind::Float, Device::Cpu))
        .index_put(&index, &Tensor::of_slice(&[1f32]), false)
}

pub struct NN {
    vs: VarStore,
    pool_size: i64,
    layer_stack1: Vec<Conv2D>,
    layer_stack2: Vec<Conv2D>,
    output_layer: Conv2D,
    opt: Optimizer,
}

fn conv2d(vs: &VarStore, inputs: usize, outputs: usize) -> Conv2D {
    let config = ConvConfig {padding: 1, .. ConvConfig::default()};

    nn::conv2d(vs.root(), inputs as i64, outputs as i64, 3, config)
}

impl NN {
    pub fn new(
        pool_size: usize,
        layer_sizes1: &[usize],
        layer_sizes2: &[usize],
        inputs: usize,
        outputs: usize,
        rate: f64
    ) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());

        let mut layer_stack1 = Vec::new();
        let mut layer_stack2 = Vec::new();

        let out1 = *layer_sizes1.last().unwrap();
        let out2 = *layer_sizes2.last().unwrap();

        layer_stack1.push(conv2d(&vs, inputs, layer_sizes1[0]));

        for sizes in layer_sizes1.windows(2) {
            layer_stack1.push(conv2d(&vs, sizes[0], sizes[1]));
        }

        layer_stack2.push(conv2d(&vs, out1, layer_sizes2[0]));

        for sizes in layer_sizes2.windows(2) {
            layer_stack2.push(conv2d(&vs, sizes[0], sizes[1]));
        }

        let output_layer = conv2d(&vs, out1 + out2, outputs);
        let opt = Adam::default().build(&vs, rate).unwrap();

        Self {
            vs,
            pool_size: pool_size as i64,
            layer_stack1,
            layer_stack2,
            output_layer,
            opt,
        }
    }

    pub fn to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.pool_size.to_be_bytes())?;

        let ctx = "Layer has no bias to write!";
        let mut named_tensors = Vec::new();

        for (i, layer) in self.layer_stack1.iter().enumerate() {
            named_tensors.push((format!("1w{}", i), &layer.ws));
            named_tensors.push((format!("1b{}", i), layer.bs.as_ref().context(ctx)?));
        }

        for (i, layer) in self.layer_stack2.iter().enumerate() {
            named_tensors.push((format!("2w{}", i), &layer.ws));
            named_tensors.push((format!("2b{}", i), layer.bs.as_ref().context(ctx)?));
        }

        named_tensors.push(("ow".to_string(), &self.output_layer.ws));
        named_tensors.push(("ob".to_string(), self.output_layer.bs.as_ref().context(ctx)?));

        Tensor::save_multi_to_stream(&named_tensors, writer)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R, rate: f64) -> Result<Self> {
        let mut buf = [0; 8];

        reader.read_exact(&mut buf)?;

        let pool_size = i64::from_be_bytes(buf);
        let vs = VarStore::new(Device::cuda_if_available());

        let mut buf = Vec::new();

        reader.read_to_end(&mut buf)?;

        let named_tensors = Tensor::load_multi_from_stream_with_device(Cursor::new(buf), vs.device())?
            .into_iter()
            .collect::<HashMap<_, _>>();

        let mut layer_stack1 = Vec::new();
        let mut layer_stack2 = Vec::new();

        for i in 0.. {
            if let Some(ws) = named_tensors.get(&format!("1w{}", i)) {
                let mut layer = conv2d(&vs, 1, 1);

                layer.ws = ws.copy();
                layer.bs = named_tensors.get(&format!("1b{}", i)).map(Tensor::copy);

                layer_stack1.push(layer);
            } else {
                break;
            }
        }

        for i in 0.. {
            if let Some(ws) = named_tensors.get(&format!("2w{}", i)) {
                let mut layer = conv2d(&vs, 1, 1);

                layer.ws = ws.copy();
                layer.bs = named_tensors.get(&format!("2b{}", i)).map(Tensor::copy);

                layer_stack2.push(layer);
            } else {
                break;
            }
        }

        let mut output_layer = conv2d(&vs, 1, 1);

        output_layer.ws = named_tensors.get("ow").unwrap().copy();
        output_layer.bs = named_tensors.get("ob").map(Tensor::copy);

        let opt = Adam::default().build(&vs, rate).unwrap();

        Ok(NN { vs, pool_size, layer_stack1, layer_stack2, output_layer, opt })
    }

    pub fn from_file(name: &str, rate: f64) -> Result<Self> {
        Self::from_reader(&mut File::open(name)?, rate)
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        let mut state1 = inputs.copy();

        for layer in &self.layer_stack1 {
            state1 = state1.apply(layer).leaky_relu();
        }

        let mut state2 =
            state1.max_pool2d(
                &[self.pool_size, self.pool_size],
                &[self.pool_size, self.pool_size],
                &[0,0],
                &[1,1],
                true
            );

        for layer in &self.layer_stack2 {
            state2 = state2.apply(layer).leaky_relu();
        }

        state2 = state2
            .repeat_interleave_self_int(self.pool_size, Some(2), None)
            .slice(2, None, Some(state1.size()[2]), 1)
            .repeat_interleave_self_int(self.pool_size, Some(3), None)
            .slice(3, None, Some(state1.size()[3]), 1);

        Tensor::concat(&[state1, state2], 1)
            .apply(&self.output_layer)
            .flatten(1, 3)
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.opt = Adam::default().build(&self.vs, rate).unwrap();
    }

    pub fn device(&self) -> Device {self.vs.device()}

    pub fn test(&self, features: &Tensor, expected: &Tensor) -> f64 {
        tch::no_grad(|| {
            let mut loss = 0.;
            let mut iter = Iter2::new(&features, &expected, 256);

            for (features, expected) in iter.return_smaller_last_batch().to_device(self.device()) {
                let out = self.forward(&features);
                loss += out
                    .cross_entropy_loss::<Tensor>(&expected, None, Reduction::Sum, -100, 0.)
                    .double_value(&[])
            }

            loss
        })
    }

    pub fn train(&mut self, features: &Tensor, expected: &Tensor) {
        let mut iter = Iter2::new(&features, &expected, 256);

        let iter2 = iter
            .return_smaller_last_batch()
            .to_device(self.device())
            .shuffle();

        for (features, expected) in iter2 {
            let res = self.forward(&features);
            let loss = res.cross_entropy_loss::<Tensor>(&expected, None, Reduction::Mean, -100, 0.);

            // println!("{}", i);

            // loss.print();

            self.opt.backward_step(&loss);
        }
    }
}

pub struct NNBot {
    nn: NN,
    feature_gen: FeatureGen,
}

impl NNBot {
    pub fn from_file(file: &str) -> Result<Self> {
        Ok(Self {
            nn: NN::from_file(file, 0.)?,
            feature_gen: FeatureGen::new()
        })
    }

    pub fn new(nn: NN) -> Self {
        Self {
            nn,
            feature_gen: FeatureGen::new()
        }
    }
}

impl Player for NNBot {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        self.feature_gen.update(state);
        let features = self.feature_gen.generate_features(player, self.nn.device());

        // disable gradient tracking to speed up network evaluation
        let tensor = tch::no_grad(|| self.nn.forward(&Tensor::stack(&[features], 0)));

        Some(move_of_tensor(&tensor.i(0), state.width, state.height))
    }
}
