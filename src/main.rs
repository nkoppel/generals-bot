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

    tch::maybe_init_cuda();
    tch::Cuda::set_user_enabled_cudnn(true);
    tch::Cuda::cudnn_set_benchmark(true);
    let args: Vec<String> = env::args().collect();

    // println!("{}", test_from_replays(&args[1], &NN::from_file("nets/net0_0_100.gio_nn", 0.).unwrap()).unwrap());
    // println!("{}", test_from_replays(&args[1], &NN::from_file("nets/net0_0_10000.gio_nn", 0.).unwrap()).unwrap());
    // println!("{}", test_from_replays(&args[1], &NN::from_file("nets/net0_0_22900.gio_nn", 0.).unwrap()).unwrap());

    // let bot = NNBot::from_file("net0_0_22900.gio_nn").unwrap();
    // let bot = RandomBot{};

    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.run_1v1(&mut (Box::new(SmartBot::new()) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(&mut (Box::new(bot) as Box<dyn Player>));
    // client.debug_listen();

    // let players =
        // vec![
            // Box::new(bot) as Box<dyn Player>,
            // Box::new(RandomBot{}),
        // ];

    // let mut state = State::generate(18, 18, 60, 10, 2);

    // let mut sim = Simulator::new(state, players);

    // sim.sim(100000, 0, true);

    // let (mut sim, len) = Replay::read_from_file("replays_prod/H5LmDhpUg.gioreplay").unwrap().to_simulator();

    // sim.sim(1000000, 0, true);

    println!("{} {}", tch::Cuda::is_available(), tch::Cuda::cudnn_is_available());
    let mut nn = NN::new(4, &[128, 128, 128, 128], &[128, 128, 128, 128], 11, 8, 0.0001);

    train_from_replays(&args[1], &args[2], 1, &mut nn).unwrap()
}
