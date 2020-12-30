mod state;
mod simulator;
mod bots;

use state::*;
use simulator::*;
use bots::*;

fn main() {
    let bots =
        vec![
            Box::new(RandomBot::new(0)) as Box<dyn Player>,
            Box::new(RandomBot::new(1)),
        ];

    let state = State::generate(10, 10, 20, 7, 2);
    let mut sim = Simulator::new(state, bots);

    sim.sim(500);

    // let mut state = State::generate(10, 10, 0, 0, 0);

    // state.generals.push(0);
    // state.armies[0] = 5;
    // state.cities[0] = 0;
    // state.terrain[0] = 0;

    // state.cities[10] = 0;
    // state.armies[10] = 10;
    // state.terrain[10] = -1;

    // println!("{}", state.move_is_valid(0, Move::new(0, 10, false)));
}
