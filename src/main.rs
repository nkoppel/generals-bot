#![feature(test)]
#![allow(dead_code)]

mod bots;
mod client;
mod replays;
mod simulator;
mod state;

#[allow(unused_imports)]
use crate::{bots::*, client::*, replays::*, simulator::*, state::*};

fn main() {
    // use std::env;

    // tch::maybe_init_cuda();
    // tch::Cuda::set_user_enabled_cudnn(true);
    // tch::Cuda::cudnn_set_benchmark(true);
    // let args: Vec<String> = env::args().collect();

    // let bot = RandomBot{};

    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.run_1v1(&mut (Box::new(SmartBot::new()) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(&mut (Box::new(bot1) as Box<dyn Player>));
    // client.debug_listen();

    // let players =
    // vec![
    // Box::new(bot) as Box<dyn Player>,
    // Box::new(bot),
    // ];

    // let mut state = State::generate(18, 18, 60, 10, 2, 15);

    // let mut sim = Simulator::new(state, players);

    // sim.sim(100000, 250, true);

    // let (mut sim, len) = Replay::read_from_file("replays_prod/H5LmDhpUg.gioreplay").unwrap().to_simulator();

    // sim.sim(1000000, 0, true);

    // println!("{} {}", tch::Cuda::is_available(), tch::Cuda::cudnn_is_available());
    // let mut nn = NN::new(3, &[256; 6], &[256; 6], &[64, 64, 8], 176, 0.001);
    // // let mut nn = NN::from_file("nets2/net4_0_5200.gio_nn", 0.000005).unwrap();

    // train_from_replays(&args[1], &args[2], 1, &mut nn).unwrap()
    test();
}
