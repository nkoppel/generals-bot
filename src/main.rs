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
    let args: Vec<String> = env::args().collect();
    let mut client = Client::new("ws://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    client.join_private(&args[2]);

    client.run_player(Box::new(RandomBot::new()));
    // client.debug_listen();
}
