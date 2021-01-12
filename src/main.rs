#![feature(duration_saturating_ops)]

mod state;
mod simulator;
mod bots;
mod client;

use state::*;
use simulator::*;
use bots::*;
use client::*;

use std::env;

fn main() {
    // let args: Vec<String> = env::args().collect();
    // println!("{:?}", args);
    // let mut client = Client::new("ws://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.join_private(&args[2]);
    // client.join_1v1();

    // client.run_player(Box::new(SmartBot::new()));
    // client.debug_listen();

    let players =
        vec![
            Box::new(SmartBot::new()) as Box<dyn Player>,
            Box::new(NoneBot{})
        ];

    let state = State::generate(18, 18, 60, 10, 2);

    let mut sim = Simulator::new(state, players);

    sim.sim(100000, 250);
}
