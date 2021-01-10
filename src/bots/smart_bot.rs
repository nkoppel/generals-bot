use super::*;

use std::collections::{VecDeque, HashSet};
use std::mem;
use std::time::{Duration, Instant};

enum BotMode {
    Gather_for(usize, usize, bool), // loc, time, hide
    Gather_until(usize, usize, bool), // loc, armies, hide
    Probe_from(usize), // loc
    Fast_expand,
    Expand_strand(usize, usize), // loc, time
    Opener,
    Attack(usize, usize), // loc, time
    Defend
}

use BotMode::*;

pub struct SmartBot {
    old_state: State,
    state: State,
    seen_cities: HashSet<usize>,
    seen_generals: Vec<isize>,
    seen_terrain: Vec<isize>,
    player_cities: Vec<usize>,
    player: usize,
    moves: VecDeque<(usize, usize)>,
    modes: VecDeque<BotMode>
}

impl SmartBot {
    pub fn new() -> Self {
        Self {
            old_state: State::new(),
            state: State::new(),
            seen_cities: HashSet::new(),
            seen_generals: Vec::new(),
            seen_terrain: Vec::new(),
            player_cities: Vec::new(),
            player: 0,
            moves: VecDeque::new(),
            modes: VecDeque::new(),
        }
    }

    fn update(&mut self, diff: StateDiff) {
        self.old_state = self.state.clone();

        self.state.patch(diff);

        if self.state.turn % 2 == 0 && self.state.turn % 50 != 0 {
            for i in 0..self.state.scores.len() {
                if self.state.turn % 50 == 2 {
                    self.player_cities[i] =
                        self.state.scores[i].1 - self.old_state.scores[i].1;
                } else {
                    self.player_cities[i] = self.player_cities[i].max(
                        self.state.scores[i].1 - self.old_state.scores[i].1
                    );
                }
            }
        }

        for c in &self.state.cities {
            let c = *c as usize;
            self.seen_cities.insert(c);
        }

        for i in 0..self.state.generals.len() {
            if self.seen_generals[i] == -1 {
                self.seen_generals[i] = self.old_state.generals[i];
            }
        }

        for (i, t) in self.state.terrain.iter().enumerate() {
            if *t >= 0 {
                self.seen_terrain[i] = *t;
            }
        }
    }

    fn expand_strand(&mut self, state: &mut State, start: usize, cost: usize) {
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

        self.moves.extend(tree.serialize_outwards().iter().copied());

        for t in seq {
            state.terrain[t] = self.player as isize;
        }
    }

    fn find_strand_loc(&self, dist: usize) -> Option<usize> {
        let mut map: Vec<isize> = self.state.terrain
            .iter()
            .map(|x|
                if *x == self.player as isize {
                    1
                } else if *x == -1 || *x == -3 {
                    -1
                } else {
                    0
                }
            )
            .collect();

        for c in &self.state.cities {
            if map[*c as usize] <= 0 {
                map[*c as usize] = -1;
            }
        }

        let dist_field = min_distance(self.state.width, &map);
        let t = select_rand_eq(&dist_field, &dist)?;

        Some(nearest(self.state.width, &dist_field, t))
    }

    fn opener(&mut self, start: usize) {
        let mut state = self.state.clone();

        for (wait, cost) in vec![(24, 14), (3, 9)] {
            self.moves.extend((0..wait).map(|_| (0, 0)));
            self.expand_strand(&mut state, start, cost);
        }
    }

    fn fast_land(&mut self) {
        for i in 0..self.state.armies.len() {
            if self.state.terrain[i] == self.player as isize {
                for n in get_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] != self.player as isize &&
                       self.state.terrain[n] != TILE_MOUNTAIN &&
                       i != self.state.generals[self.player] as usize &&
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

    fn gather(&mut self, mut time: usize, loc: usize, hide: bool) {
        let reward_func = |i| {
            let mut out = 0;

            if hide {
                for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] >= 0 && self.state.terrain[n] != self.player as isize {
                        out = -100;
                        break;
                    }
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

        while time > 0 && (paths[time].0 == 0 || paths[time].0 == paths[time - 1].0) {
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
        // println!("{}", self.state);

        let now = Instant::now();
        let capital = self.state.generals[self.player] as usize;

        if self.moves.is_empty() {
            if self.state.turn < 5 {
                self.opener(capital);
            } else {
                self.fast_land();

                if self.moves.is_empty() {
                    self.gather(50 - self.state.turn % 50, capital, true);
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
