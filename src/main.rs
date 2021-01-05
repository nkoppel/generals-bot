mod state;
mod simulator;
mod bots;
mod client;

use state::*;
use simulator::*;
use bots::*;
use client::*;
use bots::path_optimization::*;

use std::env;

fn main() {
    // let args: Vec<String> = env::args().collect();
    // let mut client = Client::new("ws://botws.generals.io/socket.io/?EIO=3&transport=websocket", &args[1]);

    // client.join_private(&args[2]);

    // client.run_player(Box::new(RandomBot::new()));
    // client.debug_listen();

    let mut rng = thread_rng();
    
    // let cost = vec![1; 81];
    let reward: Vec<isize> = (0..81).map(|_| rng.gen_range(0, 10)).collect();

    debug_print_reward(9, reward.clone());

    let cost_func = |_| 1;
    let reward_func = move |i| reward[i];

    let (paths, parents) = find_path(9, 9, 40, true, 9, reward_func, cost_func);

    println!("{:?}", paths);
    println!();
    debug_print_parents(9, parents.clone());

    let tree = PathTree::from_path(paths[9].clone(), parents);

    println!("{:?}", tree.serialize_inwards());
}
