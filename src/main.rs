#![feature(test)]
#![allow(dead_code)]
#![feature(trait_upcasting)]
#![feature(generic_const_exprs)]

mod bots;
mod client;
mod replays;
mod simulator;
mod state;

#[allow(unused_imports)]
use crate::{bots::*, client::*, replays::*, simulator::*, state::*};
use dfdx::prelude::*;

fn main() {
    train();

    // test::<TinyNet, Cuda>();
    // test::<SmallNet, Cuda>();
    // test::<BigNet, Cuda>();
    let dev = Cpu::default();
    let mut net1 = dev.build_module::<TinyNet, f32>();
    let mut net2 = dev.build_module::<TinyNet, f32>();
    net1.load("nets/tinynet_1_50.npz").unwrap();
    net2.load("nets/tinynet_1_50.npz").unwrap();

    let state = State::generate_1v1();

    let bot1 = NNBot::new(net1, dev.clone());
    let bot2 = NNBot::new(net2, dev);

    // let bot1 = RandomBot {};
    // let bot2 = RandomBot {};

    let players = vec![Box::new(bot1) as Box<dyn Player>, Box::new(bot2)];

    let mut sim = Simulator::new(state, players);

    sim.sim(100000, 250, true);
    // println!("{:?}", sim.state.scores);
    // println!("{}", sim.state);

    // use std::env;
    // let args: Vec<String> = env::args().collect();

    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.run_1v1(&mut (Box::new(SmartBot::new()) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(&mut (Box::new(bot1) as Box<dyn Player>));
    // client.debug_listen();
}
