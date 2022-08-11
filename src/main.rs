#![feature(test)]
#![allow(dead_code)]

mod state;
mod simulator;
mod replays;
mod bots;
mod client;

#[allow(unused_imports)]
use crate::{
    state::*,
    simulator::*,
    replays::*,
    bots::*,
    client::*,
};

fn main() {
    use std::env;

    let args: Vec<String> = env::args().collect();
    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.run_1v1(&mut (Box::new(SmartBot::new()) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(&mut (Box::new(RandomBot{}) as Box<dyn Player>));
    // client.debug_listen();

    // let players =
        // vec![
            // Box::new(RandomBot{}) as Box<dyn Player>,
            // Box::new(RandomBot{}),
            // Box::new(RandomBot{}),
        // ];

    // let mut state = State::generate(18, 18, 60, 10, 3);

    // state.teams = vec![0, 0, 1];

    // let mut sim = Simulator::new(state, players);

    // sim.sim(100000, 0, false);

    // let (mut sim, len) = Replay::read_from_file("replays_prod/H5LmDhpUg.gioreplay").unwrap().to_simulator();

    // sim.sim(1000000, 0, true);

    tch::maybe_init_cuda();
    tch::Cuda::set_user_enabled_cudnn(true);
    tch::Cuda::cudnn_set_benchmark(true);
    println!("{} {}", tch::Cuda::is_available(), tch::Cuda::cudnn_is_available());
    let mut nn = NN::new(4, &[128, 32, 32, 32], &[32, 32, 32, 32], 11, 8, 0.01);

    train_from_replays(&args[1], &args[2], 1, &mut nn).unwrap()
}
