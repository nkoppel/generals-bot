#![feature(test)]
#![allow(unused)]

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
    // let mut client = Client::new("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.run_1v1(&mut (Box::new(SmartBot::new()) as Box<dyn Player>));

    // client.join_private(&args[2]);
    // client.set_force_start(&args[2]);
    // client.run_player(Box::new(SmartBot::new()));
    // client.debug_listen();

    let players =
        vec![
<<<<<<< HEAD
            Box::new(RandomBot{}) as Box<dyn Player>,
            Box::new(RandomBot{}),
=======
            Box::new(SmartBot::new()) as Box<dyn Player>,
            Box::new(RandomBot::new())
>>>>>>> parent of bedf2da (add action system)
        ];

    let state = State::generate(18, 18, 60, 10, 2);

    let mut sim = Simulator::new(state, players);

    sim.sim(100000, 250, false);
}
