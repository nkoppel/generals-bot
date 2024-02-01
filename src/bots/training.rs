use super::*;

use dfdx::{
    data::IteratorBatchExt,
    optim::{Adam, AdamConfig, WeightDecay},
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
const POLICY_LR: f32 = 5e-4;
const VALUE_LR: f32 = 5e-4;
const WEIGHT_DECAY: f32 = 1e-4;
const PPO_STEPS: usize = 50;
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

impl State {
    fn general_army_distance(&self, player: usize) -> f32 {
        if self.generals[player] < 0 {
            return 0.0;
        }
        if self.generals[player ^ 1] < 0 {
            return (self.width + self.height) as f32 * DISTANCE_REWARD_FACTOR;
        }
        let distance = distance_field(self.width, self.height, &|i| {
            if i == self.generals[player] as usize {
                DistanceTile::Source
            } else if self.terrain[i] == TILE_MOUNTAIN {
                DistanceTile::Obstacle
            } else {
                DistanceTile::Empty
            }
        });

        let weighted_sum = distance
            .into_iter()
            .enumerate()
            .filter_map(|(i, d)| {
                (self.terrain[i] >= 0 && self.terrain[i] != player as isize)
                    .then(|| d * self.armies[i] as usize)
            })
            .sum::<usize>();

        weighted_sum as f32 / self.scores[player ^ 1].1 as f32 * DISTANCE_REWARD_FACTOR
    }

    fn general_distance(&self, player: usize) -> Option<usize> {
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

struct Agent {
    feature_gen: FeatureGen<D>,
    data: EpisodeData,
    prev_value: f32,
    dev: D,
}

impl Agent {
    fn get_features(&mut self, state: State) -> Tensor<InputShape, f32, D> {
        self.feature_gen.update(state, &self.dev);
        self.feature_gen.features()
    }

    fn update_rewards(&mut self, state: &State) {
        let mut value = state.general_army_distance(0) - state.general_army_distance(1);

        if self.feature_gen.player == 1 {
            value = -value;
        }
        let reward = value - self.prev_value;
        self.prev_value = value;

        self.data.rewards.push(reward);
    }

    fn get_move(&mut self, policy: Tensor<ActionShape, f32, D>, value: f32) -> Option<Move> {
        let cpu = Cpu::default();
        let observation = self.feature_gen.buf.as_ref().unwrap().clone();
        let mask = self.feature_gen.action_mask(&self.dev);

        let (action, mov, _prob) = if let Some((a, m, p)) = self
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

        mov
    }
}

pub fn run_games<NN: Net<D>>(net: NN, dev: &D) -> (Vec<EpisodeData>, NN) {
    let mut states = State::generate_1v1_batch(PPO_STEPS);
    let mut agents = (0..PPO_STEPS * 2)
        .map(|i| {
            let mut feature_gen = FeatureGen::new();
            feature_gen.player = i % 2;
            Agent {
                feature_gen,
                data: EpisodeData::default(),
                prev_value: 0.,
                dev: dev.clone(),
            }
        })
        .collect::<Vec<_>>();

    let width = states[0].width;
    let height = states[0].height;
    let mut completed_states = Vec::new();
    let mut completed_agents = Vec::new();

    for _ in (0..EPISODE_LENGTH).progress() {
        let observations = states
            .iter()
            .flat_map(|s| [s.get_player_state(0), s.get_player_state(1)])
            .zip(agents.iter_mut())
            .map(|(s, a)| a.get_features(s))
            .collect::<Vec<_>>()
            .stack();

        let observations = observations
            .reshape_like(&(agents.len(), Const, height, width))
            .unwrap();
        let (policies, values) = net.forward(observations);

        let moves = agents
            .iter_mut()
            .zip((0..states.len() * 2).map(|i| policies.clone().select(dev.tensor(i))))
            .zip(values.as_vec().into_iter())
            .map(|((agent, policy), value)| agent.get_move(policy, value));

        for (moves, state) in moves.batch_exact(Const::<2>).zip(states.iter_mut()) {
            state.step(&moves);
        }

        for (state, agent) in states.iter().flat_map(|s| [s; 2]).zip(agents.iter_mut()) {
            agent.update_rewards(state);
        }

        for i in (0..states.len()).rev() {
            if states[i].game_over().is_some() {
                completed_states.push(states.swap_remove(i));
                completed_agents.push(agents.swap_remove(i * 2 + 1));
                completed_agents.push(agents.swap_remove(i * 2));
                let len = completed_agents.len();
                completed_agents.swap(len - 1, len - 2);
            }
        }
    }

    states.append(&mut completed_states);
    agents.append(&mut completed_agents);

    for (i, state) in states.iter().enumerate() {
        println!(
            "{}/{PPO_STEPS} {} {} {state}",
            i + 1,
            agents[i * 2].data.rewards.iter().sum::<f32>(),
            agents[i * 2 + 1].data.rewards.iter().sum::<f32>()
        );
    }

    let data = agents
        .into_iter()
        .map(|mut a| {
            a.data.compute_advantage();
            a.data.compute_target_values();
            a.data
        })
        .collect();

    (data, net)
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

pub fn train_ppg<NN: Net<D>, Opt: Optimizer<NN, D, f32>>(
    mut nn: NN,
    policy_opt: &mut Opt,
    value_opt: &mut Opt,
    aux_opt: &mut Opt,
    dev: &D,
) {
    for epoch in 1.. {
        println!("Begin epoch {epoch}");
        let (data_group, nn2) = run_games(nn, dev);
        nn = nn2;

        for (e, episode) in data_group.clone().iter_mut().enumerate() {
            println!(
                "Begin training for episode {}/{}, epoch {epoch}",
                e + 1,
                data_group.len()
            );
            train_ppo(&mut nn, policy_opt, value_opt, episode, dev.clone());
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
                train_auxiliary(&mut nn, aux_opt, episode, dev.clone());
            }
        }

        nn.save(format!("nets/unet_7_{epoch}.npz"))
            .expect("failed to save network");
    }
}

type Built<M, D, E> = <M as BuildOnDevice<D, E>>::Built;

pub fn train() {
    let dev = D::default();
    dev.try_disable_cache().unwrap();
    let mut nn = dev.build_module::<UNet, f32>();
    // nn.load("nets/unet_6_40.npz").unwrap();
    let config = AdamConfig {
        lr: POLICY_LR,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(WEIGHT_DECAY)),
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

    train_ppg(nn, &mut policy_opt, &mut value_opt, &mut aux_opt, &dev)
}
