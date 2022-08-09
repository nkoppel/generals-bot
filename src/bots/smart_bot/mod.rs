use super::*;

use std::collections::{VecDeque, HashSet};
use std::mem;
use std::time::{Duration, Instant};

mod path_optimization;
mod actions;
mod strategy;

use path_optimization::*;
pub use strategy::*;

const FL_ATTEMPTS: usize = 10;

pub struct SmartBot {
    old_state: State,
    state: State,
    seen_cities: HashSet<usize>,
    seen_generals: Vec<isize>,
    seen_terrain: Vec<isize>,
    player_cities_tmp: Vec<usize>,
    player_cities: Vec<usize>,
    player: usize,
    previous_action: Action
}

impl SmartBot {
    pub fn new() -> Self {
        Self {
            old_state: State::new(),
            state: State::new(),
            seen_cities: HashSet::new(),
            seen_generals: Vec::new(),
            seen_terrain: Vec::new(),
            player_cities_tmp: Vec::new(),
            player_cities: Vec::new(),
            player: 0,
            previous_action: Action::empty()
        }
    }

    fn update(&mut self, diff: StateDiff) {
        self.old_state = self.state.clone();

        self.state.patch(diff);

        if self.player_cities.is_empty() {
            self.player_cities = vec![1; self.state.generals.len()];
            self.player_cities_tmp = vec![1; self.state.generals.len()];
        }

        if self.state.turn % 50 == 0 {
            self.player_cities = mem::replace(
                &mut self.player_cities_tmp,
                vec![1; self.state.generals.len()]
            );
        } else if self.state.turn % 2 == 0 {
            for i in 0..self.state.scores.len() {
                self.player_cities_tmp[i] = self.player_cities_tmp[i].max(
                    self.state.scores[i].0.saturating_sub(self.old_state.scores[i].0)
                );

                self.player_cities[i] = self.player_cities[i].max(
                    self.player_cities_tmp[i]
                );
            }
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

    fn expand(&self, start: usize, len: usize, cling: bool)
        -> Vec<(usize, usize)>
    {
        let mut reward = vec![0.; self.state.terrain.len()];

        for i in 0..self.state.terrain.len() {
            if self.state.terrain[i] == self.player as isize {
                continue;
            }

            let mut out: f64 = 0.;

            for n in get_neighbors(self.state.width, self.state.height, i) {
                if cling {
                    match self.state.terrain[n] {
                        -2 | -4 => out += 1.,
                        n if n == self.player as isize => out += 2.,
                        _ => (),
                    }
                } else {
                    match self.state.terrain[n] {
                        -1 | -3 => out += 1.,
                        n if n >= 0 && n != self.player as isize => out += 1.,
                        _ => (),
                    }
                }
            }

            reward[i] = out;
        };

        let obstacles = self.state.terrain
            .iter()
            .enumerate()
            .map(|(i, t)| *t == -2 || *t == -4 || self.state.cities.contains(&(i as isize)))
            .collect::<Vec<_>>();

        let mut pather = Pather::new(self.state.width, &reward, &obstacles);

        pather.create_graph(start, len, 1, false);
        let paths = pather.get_best_paths(start, len, false);

        let path = paths
            .into_iter()
            .rev()
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
            .unwrap();

        pather.get_moves(&path, false)
    }

    fn find_strand_loc(&self, dist: usize, attempts: usize) -> Option<usize> {
        for _ in 0..attempts {
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

            if let Some(out) = nearest_zero(self.state.width, &dist_field, t) {
                return Some(out);
            }
        }
        None
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

        nearest_zero(self.state.width, &dist, loc)
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

    fn get_probe_loc(&self, player: usize) -> Option<usize> {
        let mut candidates = 0;
        let mut probe_map = self.state.terrain.iter()
            .copied()
            .map(|t| {
                match t {
                    -2 | -4 => -1,
                    _ => 0
                }
            })
            .collect::<Vec<isize>>();

        for i in 0..self.state.terrain.len() {
            if self.state.terrain[i] != -3 {
                continue;
            }
            for n in get_neighbors(self.state.width, self.state.height, i) {
                if self.state.terrain[n] == player as isize {
                    probe_map[i] = 1;
                    candidates += 1;
                }
            }
        }

        match candidates {
            0 => return None,
            1 => return probe_map.iter().position(|x| *x == 1),
            _ => ()
        }

        let probe_dist = min_distance(self.state.width, &probe_map);
        let mut army_tiles = Vec::new();
        let mut best_tile = usize::MAX;
        let mut best_score = 0;

        for i in 0..self.state.terrain.len() {
            if self.state.terrain[i] == self.player as isize {
                if self.state.armies[i] >= 50 {
                    army_tiles.push(i);
                } else if self.state.armies[i] > best_score {
                    best_score = self.state.armies[i];
                    best_tile = i;
                }
            }
        }

        army_tiles.push(best_tile);

        let (score, out) = army_tiles.iter()
            .copied()
            .map(|i| {
                let path = nearest_zero_path(self.state.width, &probe_dist, i);
                let mut score = 0;

                if path.len() == 0 {
                    return (f64::NEG_INFINITY, i);
                }

                for t in &path {
                    if self.state.terrain[*t] == self.player as isize {
                        score += self.state.armies[*t] - 1;
                    } else {
                        score -= self.state.armies[*t] + 1;
                    }
                }

                (score as f64 / path.len() as f64, i)
            })
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())?;

        if score == f64::NEG_INFINITY {
            return None;
        } else {
            return Some(out);
        }
    }

    fn gather_expand(&self, loc: usize, gather_time: usize, cling: bool) -> Vec<(usize, usize)> {
        let (mut moves, armies) = self.gather(gather_time, loc, false, self.state.turn >= 100);

        moves.append(&mut self.expand(loc, armies as usize, cling));
        // println!("gather_expand: {:?} {}", moves, armies);
        return moves;
    }

    fn get_gather_reward(&self, hide: bool, nocapital: bool) -> Vec<f64> {
        let mut reward = vec![0.; self.state.armies.len()];

        for i in 0..self.state.armies.len() {
            let mut r = 0;

            if hide {
                for n in get_vis_neighbors(self.state.width, self.state.height, i) {
                    if self.state.terrain[n] >= 0 &&
                       self.state.terrain[n] != self.player as isize
                    {
                        r = -100;
                        break;
                    }
                }
            }

            if nocapital && i as isize == self.state.generals[self.player] {
                r -= 100000;
            }

            if self.state.terrain[i] == self.player as isize ||
                i as isize == self.state.generals[self.player]
            {
                r += self.state.armies[i] - 1;
            } else {
                r -= self.state.armies[i] + 1;
            } 

            reward[i] = r as f64;
        }

        reward
    }

    fn gather(&self, mut time: usize, loc: usize, hide: bool, nocapital: bool)
        -> (Vec<(usize, usize)>, isize)
    {
        let reward = self.get_gather_reward(hide, nocapital);
        let obstacles = self.state.terrain
            .iter()
            .map(|i| *i == -2 || *i == -4)
            .collect::<Vec<_>>();

        let mut pather = Pather::new(self.state.width, &reward, &obstacles);

        pather.create_graph(loc, time, 1, true);
        let paths = pather.get_best_paths(loc, time, true);
        let mut path = paths
            .into_iter()
            .rev()
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
            .unwrap();

        (pather.get_moves(&path, true), path.0 as isize)
    }

    fn gather_until(&self, armies: usize, loc: usize, hide: bool, nocapital: bool)
        -> Vec<(usize, usize)>
    {
        let reward = self.get_gather_reward(hide, nocapital);
        let obstacles = self.state.terrain
            .iter()
            .map(|i| *i == -2 || *i == -4)
            .collect::<Vec<_>>();

        let max_time = (self.state.width + self.state.height) * 2;
        let mut pather = Pather::new(self.state.width, &reward, &obstacles);
        let mut time = 20;

        while time <= max_time {
            pather.create_graph(loc, time, 1, true);
            let paths = pather.get_best_paths(loc, time, true);

            for i in 0..time {
                if paths[i].0 >= armies as f64 && !paths[i].1.is_empty() {
                    return pather.get_moves(&paths[i], true);
                }
            }

            time *= 2;
            pather.reset();
        }
        Vec::new()
    }
}

mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn t_gather() {
        let state = State { width: 18, height: 18, turn: 19, terrain: vec![-1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -2, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -2, -2, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -2, 1, -2, -1, -2, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1, -1, -1, -1, -1, 1, -1, -2, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -2, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -1, -1, 0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, 0, -1, -2, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -2, -1, -2, -2, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1, -2, -1, -2], armies: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 41, 0, 0, 42, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], cities: vec![259, 221, 101, 12, 222, 205, 156, 118, 202, 206], generals: vec![180, 141], scores: vec![(4, 11), (4, 11)] };

        let mut bot = SmartBot::new();
        bot.update(State::new().diff(&state));

        let (moves, reward) = bot.gather(10, 181, false, false);

        assert_eq!(moves, vec![(180, 181)]);
        assert_eq!(reward, 7);
    }
}
