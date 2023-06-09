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

fn main() {
    // test2();
    train();

    // test::<TinyNet, Cuda>();
    // test::<SmallNet, Cuda>();
    // test::<UNet, Cuda>();
    // let dev = Cpu::default();
    // let mut net1 = dev.build_module::<SmallNet, f32>();
    // let mut net2 = dev.build_module::<SmallNet, f32>();
    // net1.load("nets/smallnet_4_62.npz").unwrap();
    // net2.load("nets/smallnet_6_48.npz").unwrap();

    // let state = State::generate_1v1();

    // let bot1 = NNBot::new(net1, dev.clone());
    // let bot2 = NNBot::new(net2, dev);

    // let players = vec![Box::new(bot1) as Box<dyn Player>, Box::new(bot2)];

    // let mut sim = Simulator::new(state, players);

    // sim.sim(100000, 250, true);
    // println!("{:?}", sim.state.scores);
    // println!("{}", sim.state);

    // use std::env;
    // let args: Vec<String> = env::args().collect();

    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // let dev = Cpu::default();
    // let mut net = dev.build_module::<SmallNet, f32>();
    // net.load("nets/smallnet_6_48.npz").unwrap();
    // let bot = NNBot::new(net, dev.clone());

    // // client.run_1v1(&mut (Box::new(bot) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(&mut (Box::new(bot) as Box<dyn Player>));
    // client.debug_listen();
}
