use super::*;

use std::collections::{VecDeque, HashSet};
use std::mem;
use std::time::{Duration, Instant};

mod path_optimization;
mod strategy;

use path_optimization::*;
pub use strategy::*;

const PROBE_LOOKAHEAD: usize = 10;
const PROBE_DIST_WEIGHT: isize = 20;
const PROBE_CONV_WEIGHT: isize = 16;

const FL_ATTEMPTS: usize = 10;

#[derive(Clone, Copy, Debug)]
enum BotMode {
    Adj_land,
    Gather_for(usize, usize, bool, bool), // time, loc, hide, nocapital
    Gather_until(usize, usize, bool, bool), // armies, loc, hide, nocapital
    Probe_from(usize, usize), // loc, player
    Expand_strand(usize, usize), // loc, time
    Wait(usize) // time
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

        if self.state.turn % 2 == 0 &&
           self.state.turn % 50 != 0 &&
           !self.player_cities.is_empty()
        {
            for i in 0..self.state.scores.len() {
                if self.old_state.scores[i].1 <= self.state.scores[i].1 {
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
        } else if self.player_cities.is_empty() {
            self.player_cities = vec![1; self.state.generals.len()];
        }

        for c in &self.state.cities {
            let c = *c as usize;
            self.seen_cities.insert(c);
        }

        if !self.seen_generals.is_empty() {
            for i in 0..self.state.generals.len() {
                if self.seen_generals[i] == -1 {
                    self.seen_generals[i] = self.old_state.generals[i];
                }
            }
        } else {
            self.seen_generals = self.state.generals.clone();
        }

        if !self.seen_terrain.is_empty() {
            for (i, t) in self.state.terrain.iter().enumerate() {
                if *t >= 0 {
                    self.seen_terrain[i] = *t;
                }
            }
        } else {
            self.seen_terrain = self.state.terrain.clone();
        }
    }

    fn expand_strand(&self, start: usize, cost: usize)
        -> Vec<(usize, usize)>
    {
        let reward_func = |i| {
            if self.state.terrain[i] == self.player as isize {
                return 0;
            }

            let mut out: isize = 0;

            for n in get_neighbors(self.state.width, self.state.height, i) {
                match self.state.terrain[n] {
                    -1 | -3 => out += 1,
                    n if n >= 0 && n != self.player as isize => out += 1,
                    _ => (),
                }
            }

            if out <= 2 {
                2
            } else {
                out
            }
        };

        let obstacles = self.state.terrain
            .iter()
            .enumerate()
            .map(|(i, t)| *t == -2 || *t == -4 || self.state.cities.contains(&(i as isize)))
            .collect::<Vec<_>>();

        let (paths, parents) =
            find_path(self.state.width, start, false, cost, reward_func, &obstacles);

        mem::drop(reward_func);

        if paths[cost - 1].1.is_empty() {
            return Vec::new();
        }

        let tree = PathTree::from_path(&paths[cost - 1], &parents);

        tree.serialize_outwards().into_iter().collect()
    }

    fn find_strand_loc(&self, dist: usize) -> Option<usize> {
        let mut map: Vec<isize> = self.state.terrain
            .iter()
            .map(|x|
                if *x == self.player as isize {
                    1
                } else if *x == -2 || *x == -4 {
                    -1
                } else {
                    0
                }
            )
            .collect();


        for c in &self.state.cities {
            if map[*c as usize] < 0 {
                map[*c as usize] = -1;
            }
        }

        let dist_field = min_distance(self.state.width, &map);
        let t = select_rand_eq(&dist_field, &dist)?;

        nearest(self.state.width, &dist_field, t)
    }

    fn find_probe_loc(&self, player: usize) -> Option<usize> {
        let mut map: Vec<isize> = self.state.terrain
            .iter()
            .enumerate()
            .map(|(i, x)|
                if *x == self.player as isize {
                    for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                        if self.state.terrain[n] == player as isize {
                            return 0;
                        }
                    }

                    1
                } else if *x == -2 || *x == -4 {
                    -1
                } else {
                    0
                }
            )
            .collect();

        for c in &self.state.cities {
            if map[*c as usize] < 0 {
                map[*c as usize] = -1;
            }
        }

        let dist_field = min_distance(self.state.width, &map);
        let t = select_rand_eq(&self.seen_terrain, &(player as isize))?;

        nearest(self.state.width, &dist_field, t)
    }

    fn find_nearest_city(&self, loc: usize) -> Option<usize> {
        let mut map: Vec<isize> = self.state.terrain
            .iter()
            .map(|x| if *x == -2 || *x == -4 { -1 } else { 0 })
            .collect();

        for c in &self.state.cities {
            if self.state.terrain[*c as usize] != self.player as isize {
                map[*c as usize] = 1;
            }
        }

        let dist = min_distance(self.state.width, &map);

        nearest(self.state.width, &dist, loc)
    } 

    fn get_player_dist(&self, player: usize) -> Vec<usize> {
        let map = self.seen_terrain
            .iter()
            .map(|x| {
                if *x == -2 || *x == -4 {
                    -1
                } else if *x == -3 {
                    0
                } else if *x == player as isize {
                    1
                } else {
                    -1
                }
            })
            .collect::<Vec<isize>>();

        min_distance(self.state.width, &map)
    }

    fn get_seen_dist(&self) -> Vec<usize> {
        let map = self.seen_terrain
            .iter()
            .map(|x| {
                if *x == -2 || *x == -4 {
                    -1
                } else if *x == -3 {
                    0
                } else {
                    1
                }
            })
            .collect::<Vec<isize>>();

        min_distance(self.state.width, &map)
    }

    fn probe(&self, loc: usize, player: usize) -> Option<(usize, usize)> {
        if self.state.terrain[loc] != self.player as isize {
            return None;
        }

        let rew = self.get_player_dist(player)
            .into_iter()
            .zip(self.get_seen_dist().into_iter())
            .zip(self.state.armies.iter())
            .zip(self.state.terrain.iter())
            .map(|(((player_dist, seen_dist), armies), t)| {
                if player_dist > PROBE_LOOKAHEAD {
                    if *t == self.player as isize {
                        -10
                    } else {
                        -armies
                    }
                } else {
                    PROBE_DIST_WEIGHT *
                    (2 * seen_dist as isize - player_dist as isize) - armies
                }
            })
            .collect::<Vec<_>>();

        let reward = |i| {
            let mut out = rew[i] * PROBE_CONV_WEIGHT;

            for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                out += rew[n];
            }

            out
        };

        let obstacles = self.state.terrain
            .iter()
            .map(|i| *i == -2 || *i == -4)
            .collect::<Vec<_>>();

        let (paths, parents) =
            find_path(self.state.width, loc, false, PROBE_LOOKAHEAD, reward, &obstacles);

        let mut best = paths.len() - 1;

        for i in 0..paths.len() - 1 {
            if paths[i].0 > paths[best].0 {
                best = i;
            }
        }

        if paths[best].1.is_empty() {
            return None;
        }

        let tree = PathTree::from_path(&paths[best], &parents);

        if let Some(mov) = tree.serialize_outwards().first() {
            if self.state.armies[mov.0] + 1 > self.state.armies[mov.1] {
                return Some(*mov);
            }
        }

        None
    }

    fn fast_land(&mut self, time: usize) {
        for _ in 0..FL_ATTEMPTS {
            if let Some(loc) = self.find_strand_loc(5) {
                let (paths, _, _) = self.gather(time - 1, loc, false, self.state.turn >= 100);

                for (i, path) in paths.iter().enumerate() {
                    if path.0 >= 0 &&
                       path.0 as usize >= time - i - 1 &&
                       !path.1.is_empty()
                    {
                        self.modes.push_back(Gather_for(i - 1, loc, false, self.state.turn >= 100));
                        self.modes.push_back(Expand_strand(loc, time - i - 1));

                        return;
                    }
                }
            }
        }
    }

    fn losing_on_cities(&self) -> bool {
        let my_cities = self.player_cities[self.player];
        print!("{} {:?}", my_cities, self.player_cities);

        if self.state.turn % 50 > 25 {
            for c in &self.player_cities {
                if *c > my_cities {
                    println!(" {}", true);
                    return true;
                }
            }
        }
        println!(" {}", false);
        false
    }

    fn get_city(&mut self) {
        let capital = self.state.generals[self.player] as usize;

        if let Some(loc) = self.find_nearest_city(capital) {
            println!("CITY {}", loc);
            self.modes.push_back(Gather_until(1, loc, false, false));
        } else {
            println!("LAND");
            self.fast_land(20);
        }
    }

    fn opener(&mut self, start: usize) {
        for (wait, cost) in vec![(23, 14), (3, 9)] {
            self.modes.push_back(Wait(wait));
            self.modes.push_back(Expand_strand(start, cost));
        }
        self.modes.push_back(Wait(1));
    }

    fn adj_land(&self) -> Option<(usize, usize)> {
        for i in 0..self.state.armies.len() {
            if self.state.terrain[i] == self.player as isize {
                for n in get_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] != self.player as isize &&
                       self.state.terrain[n] != TILE_MOUNTAIN &&
                       i != self.state.generals[self.player] as usize &&
                       self.state.armies[i] > self.state.armies[n] + 1
                    {
                        return Some((i, n))
                    }
                }
            }
        }
        None
    }

    fn gather(&self, mut time: usize, loc: usize, hide: bool, nocapital: bool)
        -> (Paths, Vec<usize>, Option<(usize, usize)>)
    {
        let reward_func = |i| {
            let mut out = 0;

            if hide {
                for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] >= 0 &&
                       self.state.terrain[n] != self.player as isize
                    {
                        out = -100;
                        break;
                    }
                }
            }

            if nocapital && i as isize == self.state.generals[self.player] {
                out -= 100000;
            }

            if self.state.terrain[i] == self.player as isize {
                out + self.state.armies[i] - 1
            } else {
                out - self.state.armies[i]
            } 
        };

        let obstacles = self.state.terrain
            .iter()
            .map(|i| *i == -2 || *i == -4)
            .collect::<Vec<_>>();

        let (paths, parents) =
            find_path(self.state.width, loc, true, time, reward_func, &obstacles);

        time -= 1;

        while time > 0 && (paths[time].0 == 0 || paths[time].0 == paths[time - 1].0) {
            time -= 1;
        }

        let mut out_move = None;

        if !paths[time].1.is_empty() {
            let mut tree = PathTree::from_path(&paths[time], &parents);

            tree.apply_priority(&reward_func);

            out_move = tree.serialize_inwards().into_iter().next();
        }
        (paths, parents, out_move)
    }

    fn gather_until(&self, armies: usize, loc: usize, hide: bool, nocapital: bool)
        -> Option<(usize, usize)>
    {
        let max_time = (self.state.width + self.state.height) * 2;

        let mut time = 1;

        while time < max_time {
            let (paths, parents, _) = self.gather(time, loc, hide, nocapital);

            for i in time / 2 .. time {
                if paths[i].0 >= armies as isize && !paths[i].1.is_empty() {
                    let tree = PathTree::from_path(&paths[i], &parents);

                    return tree.serialize_inwards().into_iter().next();
                }
            }

            time *= 2;
        }
        None
    }

    fn run_modes(&mut self) -> Option<Move> {
        if let Some(mov) = self.moves.pop_front() {
            Some(Move::tup(mov))
        } else if let Some(mode) = self.modes.pop_front() {
            println!("{:?}", mode);
            match mode {
                Adj_land => Move::opt(self.adj_land()),
                Gather_for(time, loc, hide, nocapital) => {
                    if time == 0 {
                        return None;
                    }

                    let (_, _, mov) = self.gather(time, loc, hide, nocapital);

                    self.modes.push_front(
                        Gather_for(time - 1, loc, hide, nocapital)
                    );

                    Move::opt(mov)
                }
                Gather_until(armies, loc, hide, nocapital) => {
                    let mov = self.gather_until(armies, loc, hide, nocapital);

                    if self.state.armies[loc] < armies as isize {
                        self.modes.push_front(
                            Gather_until(armies, loc, hide, nocapital)
                        );
                    }

                    Move::opt(mov)
                }
                Expand_strand(loc, time) => {
                    let moves = self.expand_strand(loc, time);

                    if moves.is_empty() {
                        None
                    } else {
                        self.moves.extend(moves[1..].iter().copied());

                        Some(Move::tup(moves[0]))
                    }
                }
                Wait(time) => {
                    if time > 0 {
                        self.modes.push_front(Wait(time - 1));
                    }

                    Some(Move::new(0, 0, false))
                }
                Probe_from(loc, player) => {
                    if let Some(mov) = self.probe(loc, player) {
                        self.modes.push_front(Probe_from(mov.1, player));

                        Some(Move::tup(mov))
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        }
    }
}
