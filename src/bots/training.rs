use std::any::Any;

use super::*;

use dfdx::{
    data::IteratorBatchExt,
    optim::{Adam, AdamConfig},
    prelude::*,
};
use indicatif::ProgressIterator;

type D = Cuda;

const EPISODE_LENGTH: usize = 1000;
const ARMY_REWARD_FACTOR: f32 = 0;
const LAND_REWARD_FACTOR: f32 = 2e-2;
const SPECTATE: bool = false;
const TD_STEP: usize = 100;
const BATCH_SIZE: usize = 167;

const B_ENTROPY: f32 = 0.01;
const B_CLONE: f32 = 1.0;
const POLICY_LR: f32 = 5e-4;
const VALUE_LR: f32 = 5e-4;
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
    // don't train off of the last step, because winning causes too large of a policy update
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

impl<NN: Net<D>> Player for TrainingBot<NN> {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        let cpu = Cpu::default();
        let reward = {
            let prev_scores = &self.feature_gen.state.scores;

            let (land1, army1) = state.scores[player];
            let (land2, army2) = state.scores[player ^ 1];
            let (prev_land1, prev_army1) = prev_scores.get(player).unwrap_or(&(0, 0));
            let (prev_land2, prev_army2) = prev_scores.get(player ^ 1).unwrap_or(&(0, 0));

            ((army1 as f32 - army2 as f32) - (*prev_army1 as f32 - *prev_army2 as f32))
                * ARMY_REWARD_FACTOR
                + ((land1 as f32 - land2 as f32) - (*prev_land1 as f32 - *prev_land2 as f32))
                    * LAND_REWARD_FACTOR
            // (land1 as f32 - *prev_land1 as f32) * LAND_REWARD_FACTOR
            // (army1 as f32 - *prev_army1 as f32) * ARMY_REWARD_FACTOR
            // + (land1 as f32 - *prev_land1 as f32) * LAND_REWARD_FACTOR
        };

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
        self.data.rewards.push(reward);
        self.data.values.push(value);

        if SPECTATE {
            println!("Player {player}: {mov:?} {prob} {value}")
        }

        mov
    }
}

pub fn run_game<NN: Net<D>>(state: State, nets: [NN; 2], dev: &D) -> [EpisodeData; 2] {
    let players = nets
        .into_iter()
        .map(|net| {
            Box::new(TrainingBot {
                data: EpisodeData::default(),
                feature_gen: FeatureGen::default(),
                net,
                dev: dev.clone(),
            }) as Box<dyn Player>
        })
        .collect();

    let mut sim = Simulator::new(state, players);

    let game_result = sim.sim(EPISODE_LENGTH, 0, SPECTATE);

    let mut data: [EpisodeData; 2] = sim
        .get_players()
        .into_iter()
        .map(|player| {
            (player as Box<dyn Any>)
                .downcast::<TrainingBot<NN>>()
                .unwrap()
                .data
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // if let Some(winner) = game_result {
        // let loser = winner ^ 1;
        // let winning_reward = 1.; // 50 land / 500 armies

        // *data[winner].rewards.last_mut().unwrap() += winning_reward;
        // *data[loser].rewards.last_mut().unwrap() -= winning_reward;
    // }

    println!("{}", sim.state);
    println!(
        "Game finished in {:?} steps with scores {:?}, rewards [{}, {}]",
        sim.state.turn,
        sim.state.scores,
        data[0].rewards.iter().sum::<f32>(),
        data[1].rewards.iter().sum::<f32>()
    );

    for d in data.iter_mut() {
        d.compute_advantage();
        d.compute_target_values();
    }

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
        let entropy_obj = entropy_obj.nans_to(0.0).negate().sum::<_, Axes2<0, 1>>();

        let ratio =
            (new_policy.select(batch.actions.clone()) - old_policy.select(batch.actions)).exp();
        let clipped_ratio = ratio.with_empty_tape().clamp(0.8, 1.2);
        let surr_obj = -(ratio * batch.advantages.clone())
            .minimum(clipped_ratio * batch.advantages)
            .sum();

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

    for batch in data.clone().into_minibatches(BATCH_SIZE, dev).progress() {
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

        let bc_obj = ((old_policy.retaped::<OwnedTape<_, _>>() - new_policy) * old_policy.exp())
            .sum::<_, Axes2<0, 1>>();
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
    for epoch in 10.. {
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

        nn.save(format!("nets/smallnet_3_{epoch}.npz"))
            .expect("failed to save network");
    }
}

type Built<M, D, E> = <M as BuildOnDevice<D, E>>::Built;

pub fn train() {
    type Dev = Cuda;

    let dev = Dev::default();
    let mut nn = dev.build_module::<SmallNet, f32>();
    nn.load("nets/smallnet_3_9.npz").unwrap();
    let config = AdamConfig {
        lr: POLICY_LR,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: None,
    };
    let mut policy_opt: Adam<_, _, Dev> = Adam::new(&nn, config);
    let mut value_opt: Adam<_, _, Dev> = Adam::new(
        &nn,
        AdamConfig {
            lr: VALUE_LR,
            ..config
        },
    );
    let mut aux_opt: Adam<_, _, Dev> = Adam::new(&nn, config);

    train_ppg(&mut nn, &mut policy_opt, &mut value_opt, &mut aux_opt, &dev)
}
