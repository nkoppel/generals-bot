use super::*;

use dfdx::{
    data::IteratorBatchExt,
    optim::{Adam, AdamConfig},
    prelude::*,
};
use indicatif::ProgressIterator;

type D = Cuda;

const EPISODE_LENGTH: usize = 1000;
const ARMY_REWARD_FACTOR: f32 = 1e-3;
const LAND_REWARD_FACTOR: f32 = 1e-2;
const DISTANCE_REWARD_FACTOR: f32 = 1e-1;
const SPECTATE: bool = false;
const TD_STEP: usize = 100;
const BATCH_SIZE: usize = 100;

const B_ENTROPY: f32 = 0.01;
const B_CLONE: f32 = 1.0;
const POLICY_LR: f32 = 1e-4;
const VALUE_LR: f32 = 1e-4;
const PPO_STEPS: usize = 32;
const AUX_EPOCHS: usize = 0;

#[derive(Debug, Default, Clone)]
pub struct EpisodeData {
    observations: Vec<Tensor<FrameShape, f32, Cpu>>,
    actions: Vec<Option<usize>>,
    policies: Vec<Tensor<ActionShape, f32, Cpu>>,
    masks: Vec<Tensor<ActionShape, f32, Cpu>>,
    values: Vec<f32>,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    target_values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct MiniBatch {
    observations: Tensor<InputBatchShape, f32, D>,
    actions: Tensor<BatchShape, usize, D>,
    policy: Tensor<ActionBatchShape, f32, D>,
    masks: Tensor<ActionBatchShape, f32, D>,
    advantages: Tensor<BatchShape, f32, D>,
    target_values: Tensor<BatchShape, f32, D>,
}

fn to_device<S: Shape, E: Unit, D: DeviceStorage, D2: DeviceStorage + TensorFromVec<E>>(
    t: Tensor<S, E, D>,
    dev: &D2,
) -> Tensor<S, E, D2> {
    let shape = t.shape();
    dev.tensor_from_vec(t.as_vec(), *shape)
}

impl EpisodeData {
    fn compute_advantage(&mut self) {
        self.advantages = vec![0.; self.rewards.len()];
        *self.advantages.last_mut().unwrap() =
            self.rewards.last().unwrap() - self.values.last().unwrap();

        for i in 0..self.rewards.len() - 1 {
            self.advantages[i] = self.rewards[i] + self.values[i + 1] - self.values[i];
        }
    }

    fn compute_target_values(&mut self) {
        let mut window_reward = 0.;

        self.target_values = vec![0.; self.rewards.len()];

        for i in (0..self.rewards.len()).rev() {
            let future_value = *self.values.get(i + TD_STEP).unwrap_or(&0.0);
            window_reward += self.rewards[i];
            window_reward -= *self.rewards.get(i + TD_STEP).unwrap_or(&0.0);
            self.target_values[i] = future_value + window_reward;
        }
    }

    fn get_inputs(&self, dev: &D) -> Vec<Tensor<InputShape, f32, D>> {
        let shape = self.observations[0].shape().concrete();

        let observations = std::iter::repeat(self.observations[0].clone())
            .take(NUM_FRAMES - 1)
            .chain(self.observations.iter().cloned())
            .map(|t| to_device(t, dev))
            .collect::<Vec<_>>()
            .stack()
            .reshape_like(&(
                FEATURES * (self.observations.len() + NUM_FRAMES - 1),
                shape[1],
                shape[2],
            ))
            .unwrap();

        let mut out = Vec::new();

        for i in 0..self.observations.len() {
            out.push(
                observations
                    .clone()
                    .slice((i * FEATURES..(i + NUM_FRAMES) * FEATURES, .., ..))
                    .realize()
                    .unwrap(),
            );
        }

        out
    }

    fn into_minibatches(
        self,
        size: usize,
        dev: D,
    ) -> impl Iterator<Item = MiniBatch> + ExactSizeIterator {
        let observations = self.get_inputs(&dev);

        observations
            .into_iter()
            .zip(self.actions.into_iter())
            .zip(self.policies.into_iter())
            .zip(self.masks.into_iter())
            .zip(self.advantages.into_iter())
            .zip(self.target_values.into_iter())
            .filter_map(
                |(((((observation, action), policy), mask), advantage), target_values)| {
                    Some((
                        ((((observation, action?), policy), mask), advantage),
                        target_values,
                    ))
                },
            )
            // small hack to implement ExactSizeIterator
            .collect::<Vec<_>>()
            .into_iter()
            .batch_with_last(size)
            .map(move |vec| {
                #[allow(clippy::type_complexity)]
                let (((((observations, actions), policies), masks), advantages), target_values): (
                    ((((Vec<_>, _), Vec<_>), Vec<_>), _),
                    Vec<_>,
                ) = vec.into_iter().unzip();

                let len = observations.len();

                MiniBatch {
                    observations: to_device(observations.stack(), &dev),
                    actions: dev.tensor_from_vec(actions, (len,)),
                    policy: to_device(policies.stack(), &dev),
                    masks: to_device(masks.stack(), &dev),
                    advantages: dev.tensor_from_vec(advantages, (len,)),
                    target_values: dev.tensor_from_vec(target_values, (len,)),
                }
            })
    }
}

pub struct TrainingBot<NN: Net<D>> {
    data: EpisodeData,
    feature_gen: FeatureGen<D>,
    net: NN,
    dev: D,
}

impl<NN: Net<D>> std::fmt::Debug for TrainingBot<NN> {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<NN: Net<D>> Player for TrainingBot<NN> {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        let cpu = Cpu::default();

        self.feature_gen.player = player;
        self.feature_gen
            .update(state.get_player_state(player), &self.dev);

        let observation = self.feature_gen.get_features();
        let (policy, value) = self.net.forward(observation.clone());
        let value = value.as_vec()[0];

        let mask = self.feature_gen.action_mask(&self.dev);

        let (action, mov, prob) = if let Some((a, m, p)) = self
            .feature_gen
            .get_move_with_mask(policy.clone(), mask.clone())
        {
            (Some(a), Some(m), p)
        } else {
            (None, None, 1.0)
        };

        let current_frame = observation
            .slice((CHANNELS - FEATURES.., .., ..))
            .realize()
            .unwrap();

        self.data.observations.push(to_device(current_frame, &cpu));
        self.data.actions.push(action);
        self.data.policies.push(to_device(policy, &cpu));
        self.data.masks.push(to_device(mask, &cpu));
        self.data.values.push(value);

        if SPECTATE {
            println!("Player {player}: {mov:?} {prob} {value}")
        }

        mov
    }
}

impl State {
    fn get_general_distance(&self, player: usize) -> Option<usize> {
        min_distance(self.width, self.height, &|i| {
            if i == self.generals[player] as usize {
                DistanceTile::Source
            } else if self.terrain[i] == TILE_MOUNTAIN {
                DistanceTile::Obstacle
            } else if self.terrain[i] >= 0 && self.terrain[i] != player as isize {
                DistanceTile::Dest
            } else {
                DistanceTile::Empty
            }
        })
    }
}

pub fn run_game<NN: Net<D>>(mut state: State, nets: [NN; 2], dev: &D) -> [EpisodeData; 2] {
    let mut players: [TrainingBot<NN>; 2] = nets
        .into_iter()
        .map(|net| TrainingBot {
            data: EpisodeData::default(),
            feature_gen: FeatureGen::default(),
            net,
            dev: dev.clone(),
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let initial_distance = state.get_general_distance(0).unwrap() as f32;
    let mut game_result = None;
    let mut rewards = Vec::new();
    let mut prev_value = 0.0;

    for _ in (0..EPISODE_LENGTH).progress() {
        let moves: Vec<Option<Move>> = players
            .iter_mut()
            .enumerate()
            .map(|(i, player)| player.get_move(&state.get_player_state(i), i))
            .collect();

        state.step(&moves);

        let value = if let Some(winner) = state.game_over() {
            game_result = Some(winner);
            initial_distance * if winner == 0 { 1.0 } else { -1.0 }
        } else {
            state.get_general_distance(0).unwrap() as f32
                - state.get_general_distance(1).unwrap() as f32
        };

        rewards.push((value - prev_value) * DISTANCE_REWARD_FACTOR);
        prev_value = value;

        if game_result.is_some() {
            break;
        }
    }

    let mut data: [EpisodeData; 2] = players
        .into_iter()
        .map(|x| x.data)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    data[0].rewards = rewards.clone();
    data[1].rewards = rewards.into_iter().map(|x| -x).collect();

    for d in data.iter_mut() {
        d.compute_advantage();
        d.compute_target_values();
    }

    println!("{}", state);
    println!(
        "Game finished in {:?} steps with scores {:?}, rewards [{}, {}]",
        state.turn,
        state.scores,
        data[0].rewards.iter().sum::<f32>(),
        data[1].rewards.iter().sum::<f32>()
    );

    data
}

// NOTE: policy and value optimzation intentionally kept seperate in this function
pub fn train_ppo<NN: Net<D>, Opt: Optimizer<NN, D, f32>>(
    nn: &mut NN,
    policy_opt: &mut Opt,
    value_opt: &mut Opt,
    data: &mut EpisodeData,
    dev: D,
) {
    let mut new_value = Vec::new();

    for batch in data
        .clone()
        .into_minibatches(BATCH_SIZE, dev.clone())
        .progress()
    {
        let (policy, value) = nn.forward_mut(batch.observations.retaped());
        new_value.extend(value.as_vec().into_iter());

        if policy.as_vec().into_iter().any(f32::is_nan) {
            debug_print_tensor(&policy.retaped().select(dev.tensor(batch.policy.shape().concrete()[0] - 1)));
            println!("{:?}", policy.shape().concrete());
            panic!("Oops! all NaNs!");
        }

        let shape = policy.shape().concrete();
        let new_shape = (shape[0], shape[1..].iter().product::<usize>());

        let new_policy = (policy + batch.masks.clone())
            .reshape_like(&new_shape)
            .unwrap()
            .log_softmax::<Axis<1>>();
        let old_policy = (batch.policy + batch.masks.clone())
            .reshape_like(&new_shape)
            .unwrap()
            .log_softmax::<Axis<1>>();

        // convert probabilities of 0.0 to 1.0 to make entropy math work nicely
        let tmp_policy = new_policy.with_empty_tape().scalar_lt(-1e10).choose(
            dev.zeros_like(&new_shape).leaky_trace(),
            new_policy.with_empty_tape(),
        );
        let entropy_obj = tmp_policy.with_empty_tape().exp() * tmp_policy.with_empty_tape();
        let entropy_obj = entropy_obj.nans_to(0.0).sum::<_, Axes2<0, 1>>();

        let ratio =
            (new_policy.select(batch.actions.clone()) - old_policy.select(batch.actions)).exp();

        let clipped_ratio = ratio.with_empty_tape().clamp(0.8, 1.2);
        let surr_obj = (ratio * batch.advantages.clone())
            .minimum(clipped_ratio * batch.advantages)
            .sum()
            .negate();

        let policy_loss = surr_obj + entropy_obj * B_ENTROPY;
        let _ = policy_opt.update(nn, &policy_loss.backward());

        let value_loss = mse_loss(value.sum::<_, Axis<1>>(), batch.target_values);
        let _ = value_opt.update(nn, &value_loss.backward());
    }

    data.values = new_value;
    data.compute_target_values();
}

pub fn train_auxiliary<NN: Net<D>, Opt: Optimizer<NN, D, f32>>(
    nn: &mut NN,
    opt: &mut Opt,
    data: &mut EpisodeData,
    dev: D,
) {
    let mut new_value = Vec::new();

    for batch in data
        .clone()
        .into_minibatches(BATCH_SIZE, dev.clone())
        .progress()
    {
        let (policy, value) = nn.forward_mut(batch.observations.retaped());
        new_value.extend(value.as_vec().into_iter());

        let shape = policy.shape().concrete();
        let new_shape = (shape[0], shape[1..].iter().product::<usize>());

        let new_policy = (policy + batch.masks.clone())
            .reshape_like(&new_shape)
            .unwrap()
            .log_softmax::<Axis<1>>();
        let old_policy = (batch.policy + batch.masks)
            .reshape_like(&new_shape)
            .unwrap()
            .log_softmax::<Axis<1>>();

        let new_policy = new_policy.with_empty_tape().scalar_lt(-1e10).choose(
            dev.zeros_like(&new_shape).leaky_trace(),
            new_policy.with_empty_tape(),
        );
        let old_policy = old_policy.with_empty_tape().scalar_lt(-1e10).choose(
            dev.zeros_like(&new_shape).leaky_trace(),
            old_policy.with_empty_tape(),
        );

        let bc_obj = ((old_policy.retaped::<OwnedTape<_, _>>() - new_policy) * old_policy.exp())
            .sum::<(), _>();
        println!("{:?}", bc_obj.as_vec());
        let value_obj = mse_loss(value.sum::<_, Axis<1>>(), batch.target_values);
        let loss = value_obj + bc_obj * B_CLONE;
        let _ = opt.update(nn, &loss.backward());
    }

    data.values = new_value;
    data.compute_target_values();
}

pub fn train_ppg<NN: Net<D> + Clone, Opt: Optimizer<NN, D, f32>>(
    nn: &mut NN,
    policy_opt: &mut Opt,
    value_opt: &mut Opt,
    aux_opt: &mut Opt,
    dev: &D,
) {
    for epoch in 1.. {
        println!("Begin epoch {epoch}");
        let mut data_group = Vec::new();

        for game in 1..=PPO_STEPS {
            println!("Begin game {game}/{PPO_STEPS}");
            let [d1, d2] = run_game(State::generate_1v1(), [nn.clone(), nn.clone()], dev);

            data_group.push(d1);
            data_group.push(d2);
        }

        for (e, episode) in data_group.clone().iter_mut().enumerate() {
            println!(
                "Begin training for episode {}/{}, epoch {epoch}",
                e + 1,
                data_group.len()
            );
            train_ppo(nn, policy_opt, value_opt, episode, dev.clone());
        }

        #[allow(clippy::reversed_empty_ranges)]
        for aux_epoch in 1..=AUX_EPOCHS {
            println!("Begin aux epoch {aux_epoch}/{AUX_EPOCHS}, in epoch {epoch}");
            for (e, episode) in data_group.clone().iter_mut().enumerate() {
                println!(
                    "Train from episode {}/{}, aux epoch {aux_epoch}/{AUX_EPOCHS}, epoch {epoch}",
                    e + 1,
                    data_group.len()
                );
                train_auxiliary(nn, aux_opt, episode, dev.clone());
            }
        }

        nn.save(format!("nets/unet_1_{epoch}.npz"))
            .expect("failed to save network");
    }
}

type Built<M, D, E> = <M as BuildOnDevice<D, E>>::Built;

pub fn train() {
    let dev = D::default();
    let mut nn = dev.build_module::<UNet, f32>();
    // nn.load("nets/smallnet_6_33.npz").unwrap();
    let config = AdamConfig {
        lr: POLICY_LR,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: None,
    };
    let mut policy_opt: Adam<_, _, D> = Adam::new(&nn, config);
    let mut value_opt: Adam<_, _, D> = Adam::new(
        &nn,
        AdamConfig {
            lr: VALUE_LR,
            ..config
        },
    );
    let mut aux_opt: Adam<_, _, D> = Adam::new(&nn, config);

    train_ppg(&mut nn, &mut policy_opt, &mut value_opt, &mut aux_opt, &dev)
}
