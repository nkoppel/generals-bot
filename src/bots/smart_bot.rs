use super::*;

use std::collections::VecDeque;
use std::mem;
use std::time::{Duration, Instant};

pub struct SmartBot {
    state: State,
    seen_state: State,
    time_seen: Vec<usize>,
    player: usize,
    moves: VecDeque<(usize, usize)>
}

impl SmartBot {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            seen_state: State::new(),
            time_seen: Vec::new(),
            player: 0,
            moves: VecDeque::new()
        }
    }

    fn expand_strand(&mut self, state: &mut State, start: usize, wait: usize, cost: usize) {
        let reward_func = |i| {
            let mut out: isize = 0;

            for n in get_neighbors(state.width, state.height, i) {
                match state.terrain[n] {
                    -1 | -3 => out += 1,
                    _ => (),
                }
            }

            if out <= 2 {
                1
            } else {
                out - 1
            }
        };

        let cost_func =
            |i| match state.terrain[i] {-2 | -4 => usize::MAX, _ => 1};

        let (paths, parents) =
            find_path(state.width, state.height, start, false, cost, reward_func, cost_func);

        mem::drop(cost_func);
        mem::drop(reward_func);

        if paths[cost].1.is_empty() {
            return;
        }

        let tree = PathTree::from_path(&paths[cost], &parents);
        let seq = get_sequences(&paths[cost], &parents)
            .into_iter().next().unwrap();

        self.moves.extend((0..wait).map(|_| (0, 0)));
        self.moves.extend(tree.serialize_outwards().iter().copied());

        for t in seq {
            state.terrain[t] = self.player as isize;
        }
    }

    fn opener(&mut self, start: usize) {
        let mut state = self.state.clone();

        for (wait, cost) in vec![(24, 14), (3, 9)] {
            self.expand_strand(&mut state, start, wait, cost);
        }
    }

    fn fast_land(&mut self) {
        for i in 0..self.state.armies.len() {
            if self.state.terrain[i] == self.player as isize {
                for n in get_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] != self.player as isize &&
                       self.state.terrain[n] != TILE_MOUNTAIN &&
                       self.state.armies[i] > self.state.armies[n] + 1
                    {
                        self.moves.push_back((i, n));
                        break;
                    }
                }
            }

            if !self.moves.is_empty() {
                break;
            }
        }
    }

    fn gather(&mut self, mut time: usize, loc: usize) {
        let reward_func = |i| {
            let mut out = 0;

            for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                if self.state.terrain[n] >= 0 && self.state.terrain[n] != self.player as isize {
                    out = -100;
                    break;
                }
            }

            out +
            self.state.armies[i] *
            if self.state.terrain[i] == self.player as isize {
                1
            } else {
                -1
            } 
        };

        let cost_func =
            |i| match self.state.terrain[i] {-2 | -4 => usize::MAX, _ => 1};

        let (paths, parents) =
            find_path(self.state.width, self.state.height, loc, true, time, reward_func, cost_func);

        while time > 2 && (paths[time].0 == 0 || paths[time].0 == paths[time - 1].0) {
            time -= 1;
        }

        if !paths[time].1.is_empty() {
            let mut tree = PathTree::from_path(&paths[time], &parents);

            tree.apply_priority(&reward_func);

            if let Some(mov) = tree.serialize_inwards().first() {
                self.moves.push_back(*mov);
            }
        }
    }
}

impl Player for SmartBot {
    fn init(&mut self, _: &Vec<usize>, player: usize) {
        self.player = player;
    }

    fn get_move(&mut self, diff: StateDiff) -> Option<Move> {
        self.state.patch(diff);
        println!("{}", self.state.turn);
        println!("{}", self.state);

        let now = Instant::now();
        let capital = self.state.generals[self.player] as usize;

        if self.moves.is_empty() {
            if self.state.turn < 5 {
                self.opener(capital);
            } else {
                self.fast_land();

                if self.moves.is_empty() {
                    self.gather(50 - self.state.turn % 50, capital);
                }
            }
        }

        println!("{}", now.elapsed().as_millis());
        println!("{:?}", self.moves);

        if let Some(mov) = self.moves.pop_front() {
            Some(Move::new(mov.0, mov.1, false))
        } else {
            None
        }
    }
}
