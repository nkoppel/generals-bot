use super::*;

use dfdx::prelude::*;

const FEATURES: usize = 10;
const NUM_FRAMES: usize = 16;
const CHANNELS: usize = FEATURES * NUM_FRAMES;

const ACTIONS: usize = 4;
const OUTPUTS: usize = ACTIONS * 2;

const BATCH_SIZE: usize = 16;

type DynImage1<const A: usize> = (Const<A>, usize, usize);
type DynImage2<const A: usize, const B: usize> = (Const<A>, Const<B>, usize, usize);

type InputShape = DynImage1<CHANNELS>;
type OutputShape = DynImage1<OUTPUTS>;
type ActionShape = DynImage1<ACTIONS>;

type InputBatchShape = DynImage2<BATCH_SIZE, CHANNELS>;
type OutputBatchShape = DynImage2<BATCH_SIZE, OUTPUTS>;
type ActionBatchShape = DynImage2<BATCH_SIZE, ACTIONS>;

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

pub type BigNet = (
    Head,
    (Repeated<Block<128, 3, 1>, 1>, Downsample<128, 64>),
    (Repeated<Block<64, 5, 2>, 2>, Downsample<64, 32>),
    (Repeated<Block<32, 9, 4>, 4>, Conv2D<32, 8, 7, 1, 3>),
);

pub type SmallNet = (
    Downsample<CHANNELS, 64>,
    Block<64, 5, 2>,
    Conv2D<64, 8, 5, 1, 2>,
);

pub type TinyNet = Conv2D<CHANNELS, 8, 3, 1, 1>;

// pub fn test() {
    // let dev = Cpu::default();
    // let model = dev.build_module::<SmallNet, f32>();
    // let input = dev.zeros_like(&(Const::<CHANNELS>::default(), 20, 30));
    // let output = model.forward(input);
    // println!("{}", model.num_trainable_params());
    // println!("{:?}", output.shape().concrete());
    // println!("{:?}", output.realize::<OutputShape>().is_ok());
// }

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
    pub fn new() -> Self {
        Self {
            state: State::new(),
            seen_generals: Vec::new(),
            seen_cities: Vec::new(),
            seen_terrain: Vec::new(),
            player: 0,
            buf: None,
        }
    }

    pub fn update(&mut self, state: &State, dev: &D) {
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

        let buf = if let Some(buf) = std::mem::take(&mut self.buf) {
            let mut tmp = buf.slice((FEATURES.., .., ..));
            tmp = tmp.concat(self.generate_features(dev));
            tmp.realize().unwrap()
        } else {
            // self.generate_features(dev)
                // .broadcast_like(&(
                    // Const::<NUM_FRAMES>::default(),
                    // Const::<FEATURES>::default(),
                    // self.state.height,
                    // self.state.width,
                // ))
                // .reshape_like(&(
                    // Const::<CHANNELS>::default(),
                    // self.state.height,
                    // self.state.width,
                // ))
                // .unwrap()
            let features = self.generate_features(dev).realize::<[usize; 3]>().unwrap();
            let mut buf = features.clone();
            for _ in 0..NUM_FRAMES - 1 {
                buf = buf.concat(features.clone());
            }

            buf.realize().unwrap()
        };

        self.buf = Some(buf);
    }

    pub fn get_features(&self) -> Tensor<InputShape, f32, D> {
        self.buf.as_ref().unwrap().clone()
    }

    fn generate_features(&self, dev: &D) -> Tensor<DynImage1<FEATURES>, f32, D> {
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
        out.extend(self.state.armies.iter().map(|a| (*a as f32 + 1.).ln()));

        dev.tensor_from_vec(out, [FEATURES, self.state.height, self.state.width])
            .realize()
            .unwrap()
    }

    fn action_mask(&self, dev: &D) -> Tensor<DynImage1<4>, f32, D> {
        let size = self.state.width * self.state.height;
        let mut out = vec![f32::NEG_INFINITY; 4 * size];

        for (i, (s, a)) in self
            .state
            .terrain
            .iter()
            .zip(self.state.armies.iter())
            .enumerate()
        {
            if *s == self.player as isize && *a > 1 {
                let mask = get_neighbor_mask(self.state.width, self.state.height, i);
                let locs = get_neighbor_locs(self.state.width, self.state.height, i);

                for dir in 0..4 {
                    if mask[dir] && self.state.terrain[locs[dir]] != TILE_MOUNTAIN {
                        out[dir * size + i] = 0.;
                    }
                }
            }
        }
        dev.tensor_from_vec(
            out,
            (Default::default(), self.state.height, self.state.width),
        )
    }

    pub fn get_move(&self, t: Tensor<ActionShape, f32, D>, dev: &D) -> Option<(usize, Move)> {
        let mask = self.action_mask(dev);
        let mut t = t.slice((..4, .., ..)).realize().unwrap();
        t = t + mask;

        let size = self.state.width * self.state.height;
        sample_logits(t.reshape_like(&(size * 4,)).unwrap()).map(|move_idx| {
            let mov = Move {
                start: move_idx % size,
                end: get_neighbor_locs(self.state.width, self.state.height, move_idx % size)
                    [move_idx / size],
                is50: false,
            };

            (move_idx, mov)
        })
    }
}

fn sample_logits<D: Device<f32>>(mut t: Tensor<(usize,), f32, D>) -> Option<usize> {
    t = t.softmax();
    let mut sample: f32 = random();

    for (i, x) in t.as_vec().into_iter().enumerate() {
        sample -= x;

        if sample < 0. {
            return Some(i);
        }
    }

    return None;
}

pub struct NNBot<NN, D: Device<f32>> {
    feature_gen: FeatureGen<D>,
    nn: NN,
    dev: D,
}

impl<NN, D: Device<f32>> NNBot<NN, D> {
    pub fn new(nn: NN, dev: D) -> Self {
        Self {
            feature_gen: FeatureGen::new(),
            nn,
            dev,
        }
    }
}

impl<
        NN: Module<Tensor<InputShape, f32, D>, Output = Tensor<OutputShape, f32, D>>,
        D: Device<f32>,
    > Player for NNBot<NN, D>
{
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        self.feature_gen.player = player;

        self.feature_gen.update(state, &self.dev);
        let features = self.feature_gen.get_features();
        let output = self.nn.forward(features);
        let (_, mov) = self.feature_gen.get_move(output, &self.dev)?;

        Some(mov)
    }
}
