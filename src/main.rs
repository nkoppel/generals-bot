#![feature(test)]
#![allow(unused)]

mod state;
mod simulator;
mod replays;
mod bots;
mod client;

use state::*;
use simulator::*;
use replays::*;
use bots::*;
use client::*;

use std::env;

fn main() {
    // let args: Vec<String> = env::args().collect();
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

    // let mut sim = Replay::read_from_file("SYZZLhOLl.gioreplay").to_simulator();

    // sim.sim(1000000, 0, true);
}
