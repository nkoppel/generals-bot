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

    // client.run_player(Box::new(SmartBot::new()));
    client.debug_listen();

    // let mut rng = thread_rng();
    
    // for i in 0..1000 {
        // let reward: Vec<isize> = (0..10201).map(|_| rng.gen_range(0, 10)).collect();

        // // debug_print_reward(51, reward.clone());

        // let cost_func = |_| 1;
        // let reward_func = |i| reward[i];

        // let (paths, parents) = find_path(101, 101, 5100, true, 20, reward_func, cost_func);

        // // println!("{:?}", paths);
        // // println!();
        // // debug_print_parents(51, parents.clone());

        // let tree = PathTree::from_path(&paths[20].clone(), &parents);

        // // println!("{:?}", tree.serialize_inwards());
    // }
}
