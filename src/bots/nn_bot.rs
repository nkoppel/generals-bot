use super::*;

use dfdx::prelude::*;

#[derive(Debug, Default)]
pub struct FeatureGen<D: Device<f32>> {
    pub state: State,
    pub seen_generals: Vec<isize>,
    pub seen_cities: Vec<bool>,
    pub seen_terrain: Vec<isize>,
    pub player: usize,
    pub buf: Option<Tensor<InputShape, f32, D>>,
}

pub fn debug_print_tensor<D: Device<f32>, S: Shape<Concrete = [usize; 3]>>(t: &Tensor<S, f32, D>) {
    let [panes, height, width] = t.shape().concrete();
    let data = t.as_vec();

    for p in 0..panes {
        for h in 0..height {
            for w in 0..width {
                print!("{:5.2} ", data[(p * height + h) * width + w]);
            }
            println!();
        }
        println!();
    }
}

impl<D: Device<f32>> FeatureGen<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, state: State, dev: &D) {
        if self.seen_terrain.is_empty() {
            self.seen_terrain = state.terrain.clone();
        }
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

        self.state = state;

        let buf = if let Some(buf) = std::mem::take(&mut self.buf) {
            let mut tmp = buf.slice((FEATURES.., .., ..));
            tmp = tmp.concat(self.generate_features(dev));
            tmp.realize().unwrap()
        } else {
            let features = self.generate_features(dev).realize::<[usize; 3]>().unwrap();
            let mut buf = features.clone();
            for _ in 0..NUM_FRAMES - 1 {
                buf = buf.concat(features.clone());
            }

            buf.realize().unwrap()
        };

        self.buf = Some(buf);
    }

    pub fn features(&self) -> Tensor<InputShape, f32, D> {
        self.buf.as_ref().unwrap().clone()
    }

    fn generate_features(&self, dev: &D) -> Tensor<DynImage1<FEATURES>, f32, D> {
        let size = self.seen_terrain.len();

        // boolean fields
        let bool_iter = std::iter::empty()
            // my tiles
            .chain(self.seen_terrain.iter().map(|x| *x == self.player as isize))
            // opponent tiles
            .chain(
                self.seen_terrain
                    .iter()
                    .map(|x| *x == (self.player ^ 1) as isize),
            )
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

        // armies (all army counts are logarithmic to keep inputs from being too large)
        out.extend(self.state.armies.iter().map(|a| (*a as f32 + 1.).ln()));
        let scores = [
            self.state.scores[self.player].0,     // my total armies
            self.state.scores[self.player].1,     // my total land
            self.state.scores[self.player ^ 1].0, // opponent's total armies
            self.state.scores[self.player ^ 1].1, // opponent's total land
        ];

        out.extend(
            scores
                .into_iter()
                .flat_map(|s| std::iter::repeat((s as f32 + 1.).ln()).take(size)),
        );

        dev.tensor_from_vec(out, [FEATURES, self.state.height, self.state.width])
            .realize()
            .unwrap()
    }

    pub fn action_mask(&self, dev: &D) -> Tensor<DynImage1<4>, f32, D> {
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

    pub fn get_move_with_mask(
        &self,
        mut t: Tensor<ActionShape, f32, D>,
        mask: Tensor<ActionShape, f32, D>,
    ) -> Option<(usize, Move, f32)> {
        t = t + mask;

        let size = self.state.width * self.state.height;
        sample_logits(t.reshape_like(&(size * 4,)).unwrap()).map(|(move_idx, prob)| {
            let mov = Move {
                start: move_idx % size,
                end: get_neighbor_locs(self.state.width, self.state.height, move_idx % size)
                    [move_idx / size],
                is50: false,
            };

            (move_idx, mov, prob)
        })
    }

    pub fn get_move(&self, t: Tensor<ActionShape, f32, D>, dev: &D) -> Option<(usize, Move, f32)> {
        let mask = self.action_mask(dev);
        self.get_move_with_mask(t, mask)
    }
}

fn sample_logits<D: Device<f32>>(mut t: Tensor<(usize,), f32, D>) -> Option<(usize, f32)> {
    t = t.softmax();
    let mut sample: f32 = random();

    // println!("{}", t.as_vec().into_iter().sum::<f32>());

    for (i, x) in t.as_vec().into_iter().enumerate() {
        sample -= x;

        if sample < 0. {
            return Some((i, x));
        }
    }

    None
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

impl<NN: Net<D>, D: Device<f32>> Player for NNBot<NN, D> {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        self.feature_gen.player = player;

        self.feature_gen.update(state.clone(), &self.dev);
        let features = self.feature_gen.features();
        let (policy, _) = self.nn.forward(features);
        let (_, mov, _) = self.feature_gen.get_move(policy, &self.dev)?;

        Some(mov)
    }
}
