use super::*;

use dfdx::prelude::*;

const FEATURES: usize = 10;
const NUM_FRAMES: usize = 16;
const BATCH_SIZE: usize = 16;
const CHANNELS: usize = FEATURES * NUM_FRAMES;

type InputShape = Rank4<BATCH_SIZE, CHANNELS, 28, 30>;

type Head = (Conv2D<CHANNELS, 128, 3, 1, 1>, BatchNorm2D<128>, ReLU);

type Block<const C: usize, const D: usize, const P: usize> = (
    Residual<(
        Conv2D<C, C, D, 1, P>,
        BatchNorm2D<C>,
        ReLU,
        Conv2D<C, C, D, 1, P>,
        BatchNorm2D<C>,
    )>,
    ReLU,
);

type Downsample<const I: usize, const O: usize> = (Conv2D<I, O, 3, 1, 1>, BatchNorm2D<O>, ReLU);

pub type Net = (
    Head,
    (Repeated<Block<128, 3, 1>, 1>, Downsample<128, 64>),
    (Repeated<Block<64, 5, 2>, 2>, Downsample<64, 32>),
    (Repeated<Block<32, 9, 4>, 4>, Conv2D<32, 8, 7, 1, 3>),
);

pub type SmallNet = (
    Downsample<CHANNELS, 64>,
    Block<64, 3, 1>,
    Conv2D<64, 8, 3, 1, 1>,
);

pub fn test() {
    let dev = Cpu::default();
    let model = dev.build_module::<SmallNet, f32>();
    println!("{}", model.num_trainable_params());
}

#[derive(Debug)]
pub struct FeatureGen<D: Device<f32>> {
    state: State,
    seen_generals: Vec<isize>,
    seen_cities: Vec<bool>,
    seen_terrain: Vec<isize>,
    player: usize,
    buf: Option<Tensor<InputShape, f32, D>>,
}

impl<D: Device<f32>> FeatureGen<D> {
    pub fn new(player: usize) -> Self {
        Self {
            state: State::new(),
            seen_generals: Vec::new(),
            seen_cities: Vec::new(),
            seen_terrain: Vec::new(),
            player,
            buf: None,
        }
    }

    pub fn update(&mut self, state: &State) {
        self.seen_generals.resize(state.generals.len(), -1);
        self.seen_terrain.resize(state.terrain.len(), -1);
        self.seen_cities.resize(state.terrain.len(), false);

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

        // if self.buf.is_empty() {
        // for _ in 0..self.num_frames {
        // self.buf.push(self.generate_features());
        // }
        // } else {
        // self.buf.rotate_left(1);
        // *self.buf.last_mut().unwrap() = self.generate_features();
        // }
    }

    fn generate_features(&self, dev: &D) -> Tensor<[usize; 3], f32, D> {
        let size = self.seen_terrain.len();

        // boolean fields
        let bool_iter = std::iter::empty()
            // my tiles
            .chain(self.seen_terrain.iter().map(|x| *x == self.player as isize))
            // opponent tiles
            .chain(self.seen_terrain.iter().map(|x| *x != self.player as isize))
            // my general
            .chain((0..size).map(|i| i as isize == self.seen_generals[self.player]))
            // opponent general
            .chain((0..size).map(|i| i as isize == self.seen_generals[self.player ^ 1]))
            // currently visible tiles
            .chain(self.state.terrain.iter().map(|x| *x >= TILE_MOUNTAIN))
            // known cities
            .chain(self.seen_cities.iter().copied())
            // known mountains
            .chain(self.seen_terrain.iter().map(|x| *x == TILE_MOUNTAIN))
            // mountain/city in fog
            .chain(self.seen_terrain.iter().map(|x| *x == TILE_FOG_OBSTACLE))
            // empty tile
            .chain(
                self.seen_terrain
                    .iter()
                    .map(|x| *x == TILE_EMPTY || *x == TILE_FOG),
            );

        let mut out: Vec<f32> = bool_iter.map(|b| b as u8 as f32).collect();

        // logarithmic armies
        out.extend(self.state.armies.iter().map(|a| (*a as f32).ln()));

        dev.tensor_from_vec(out, [FEATURES, self.state.height, self.state.width])
    }

    fn action_mask(&self, dev: &D) -> Tensor<(Const<4>, usize, usize), f32, D> {
        let size = self.state.width * self.state.height;
        let mut out = vec![0.; 4 * size];

        for (i, (s, a)) in self.state.terrain.iter().zip(self.state.armies.iter()).enumerate() {
            if *s == self.player as isize && *a > 1 {
                let mask = get_neighbor_mask(self.state.width, self.state.height, i);
                let locs = get_neighbor_locs(self.state.width, self.state.height, i);

                for dir in 0..4 {
                    if mask[dir] && self.state.terrain[locs[dir]] != TILE_MOUNTAIN {
                        out[dir * size + i] = 1.;
                    }
                }
            }
        }
        dev.tensor_from_vec(out, (Default::default(), self.state.height, self.state.width))
    }
}

// pub struct NNBot<NN> {
// nn: NN,
// feature_gen: FeatureGen,
// }

// impl NNBot {
// pub fn new(nn: NN) -> Self {
// Self {
// feature_gen: FeatureGen::with_nn(0, &nn),
// nn,
// }
// }

// pub fn from_file(file: &str) -> Result<Self> {
// Ok(Self::new(NN::from_file(file, 0.)?))
// }
// }

// impl Player for NNBot {
// fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
// self.feature_gen.player = player;

// self.feature_gen.update(state);
// let features = self.feature_gen.get_features().to_device(self.nn.device());

// // disable gradient tracking to speed up network evaluation
// let tensor = tch::no_grad(|| self.nn.forward(&Tensor::stack(&, 0)));

// // println!("{:?}", move_of_tensor(&tensor.i(0), state.width, state.height));

// Some(move_of_tensor(&tensor.i(0), state.width, state.height))
// }
// }
