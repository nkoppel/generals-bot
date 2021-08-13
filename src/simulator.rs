use std::mem;
use super::state::*;

fn to_1d(width: usize, x: isize, y: isize) -> isize {
    x + y * width as isize
}

fn from_1d(width: usize, loc: isize) -> (isize, isize) {
    (loc % width as isize, loc / width as isize)
}

pub fn get_neighbors(width: usize, height: usize, loc: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let size = width * height;

    if loc >= width {
        out.push(loc - width);
    }
    if loc + width < size {
        out.push(loc + width);
    }
    if loc > 0 && loc / width == (loc - 1) / width {
        out.push(loc - 1);
    }
    if loc < size && loc / width == (loc + 1) / width {
        out.push(loc + 1);
    }

    out
}

pub fn get_vis_neighbors(width: usize, height: usize, loc: usize) -> Vec<usize> {
    let mut out = Vec::new();

    let iwidth = width as isize;
    let mut ds = vec![1, iwidth - 1, iwidth, iwidth + 1];

    ds.extend(ds.clone().iter().map(|x| -*x));

    let size = width * height;

    for d in ds {
        let loc2 = loc as isize + d;

        if loc2 >= 0 && loc2 < size as isize {
            out.push(loc2 as usize);
        }
    }

    out
}

pub fn select_rand_eq<T>(vec: &Vec<T>, item: &T) -> Option<usize>
    where T: PartialEq
{
    let mut num_eq = 0;

    for i in vec.iter() {
        if *i == *item {
            num_eq += 1;
        }
    }

    if num_eq == 0 {
        return None;
    }

    let mut rng = thread_rng();

    let n = rng.gen_range(0..num_eq) + 1;
    let mut j = 0;

    for i in 0..vec.len() {
        if vec[i] == *item {
            j += 1;
        }
        if j == n {
            return Some(i);
        }
    }

    None
}

impl State {
    pub fn update_scores(&mut self) {
        self.scores = vec![(0, 0); self.generals.len()];

        for i in 0..self.terrain.len() {
            if self.terrain[i] >= 0 {
                let player = self.terrain[i] as usize;

                self.scores[player].0 += 1;
                self.scores[player].1 += self.armies[i] as usize;
            }
        }
    }

    pub fn get_player_state(&self, player: usize) -> Self {
        let deltas = vec![
            (-1, -1), ( 0, -1), ( 1, -1),
            (-1,  0), ( 0,  0), ( 1,  0),
            (-1,  1), ( 0,  1), ( 1,  1),
        ];

        let player = player as isize;
        let mut out = self.add_fog();

        for loc in 0..self.terrain.len() {
            if self.terrain[loc] == player {
                let (x, y) = from_1d(self.width, loc as isize);

                for (xi, yi) in &deltas {
                    let x2 = x + xi;
                    let y2 = y + yi;

                    let loc2 = to_1d(self.width, x2, y2) as usize;

                    if  x2 >= 0 &&
                        y2 >= 0 &&
                        x2 < self.width  as isize &&
                        y2 < self.height as isize
                    {
                        out.terrain[loc2] = self.terrain[loc2];
                        out.armies[loc2] = self.armies[loc2];

                        let tmp = self.generals
                            .clone()
                            .iter_mut()
                            .position(|x| *x == loc2 as isize);

                        if let Some(idx) = tmp {
                            out.generals[idx] = self.generals[idx];
                        }
                    }
                }
            }
        }

        for loc in &self.generals {
            if *loc >= 0 {
                if out.terrain[*loc as usize] == TILE_FOG_OBSTACLE {
                    out.terrain[*loc as usize] = TILE_FOG
                }
            }
        }

        for loc in &self.cities {
            if out.terrain[*loc as usize] >= TILE_EMPTY {
                out.cities.push(*loc);
            }
        }

        out
    }

    pub fn incr_armies(&mut self) {
        if self.turn % 50 == 0 {
            for i in 0..self.terrain.len() {
                if self.terrain[i] >= 0 {
                    self.armies[i] += 1;
                }
            }
        } else if self.turn % 2 == 0 {
            for i in &self.cities {
                if self.terrain[*i as usize] >= 0 {
                    self.armies[*i as usize] += 1;
                }
            }

            for i in &self.generals {
                if *i >= 0 && self.terrain[*i as usize] >= 0 {
                    self.armies[*i as usize] += 1;
                }
            }
        }

        self.turn += 1;
    }

    fn capture_player(&mut self, p1: usize, p2: usize) {
        self.cities.push(self.generals[p2]);
        self.generals[p2] = -2;

        for t in self.terrain.iter_mut() {
            if *t == p2 as isize {
                *t = p1 as isize;
            }
        }
    }

    pub fn move_is_valid(&self, player: usize, mov: Move) -> bool {
        let d = (mov.start as isize - mov.end as isize).abs() as usize;
        let size = self.armies.len();
        let mut ret = false;

        ret |= d != 1 && d != self.width;
        ret |= d == 1 && mov.start / self.width != mov.end / self.width;

        ret |= mov.start >= size;
        ret |= mov.end >= size;

        if ret {
            return false;
        }

        ret |= self.terrain[mov.start] != player as isize;
        ret |= self.armies[mov.start] < 1;

        ret |= self.terrain[mov.end] < -1;

        !ret
    }

    pub fn do_move(&mut self, player: usize, mov: Move) -> bool {
        if !self.move_is_valid(player, mov) {
            return false;
        }

        let transferred =
            if mov.is50 {
                self.armies[mov.start] / 2
            } else {
                self.armies[mov.start] - 1
            };

        self.armies[mov.start] -= transferred;

        let player2 = self.terrain[mov.end];

        if player2 == player as isize {
            self.armies[mov.end] += transferred;
        } else {
            if transferred > self.armies[mov.end] {
                if player2 >= 0 && self.generals[player2 as usize] == mov.end as isize {
                    self.capture_player(player, player2 as usize);
                }

                self.terrain[mov.end] = player as isize;
                self.armies[mov.end] = transferred - self.armies[mov.end];
            } else {
                self.armies[mov.end] -= transferred;
            }
        }

        true
    }

    pub fn get_random_move(&self, player: usize) -> Option<Move> {
        let mut rng = thread_rng();

        let start = select_rand_eq(&self.terrain, &(player as isize)).unwrap();
        let neighbors = get_neighbors(self.width, self.height, start);

        if neighbors.is_empty() {
            return None;
        }

        let end = neighbors[rng.gen_range(0..neighbors.len())];

        Some(Move::new(start, end, false))
    }
}

pub trait Player {
    fn reset(&mut self) {}
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move>;
}

use std::thread;
use std::time::{Duration, Instant};

pub struct Simulator {
    state: State,
    players: Vec<Box<dyn Player>>,
}

impl Simulator {
    pub fn new(mut state: State, mut players: Vec<Box<dyn Player>>) -> Self {
        state.remove_fog();

        Self {
            state,
            players,
        }
    }

    pub fn into_state(self) -> State {
        self.state
    }

    pub fn get_players(&mut self) -> Vec<Box<dyn Player>> {
        mem::take(&mut self.players)
    }

    pub fn sim(&mut self, rounds: usize, wait: usize, predict: bool) -> Option<usize> {
        let wait = Duration::from_millis(wait as u64);

        self.state.incr_armies();
        self.state.turn -= 1;

        // println!("{}", self.state);

        for _ in 0..rounds {
            self.state.incr_armies();
            self.state.update_scores();

            let mut state = self.state.clone();
            let mut moves = Vec::new();
            let mut active_players = 0;
            let mut last_active = 0;

            let time_spent = Instant::now();

            for player in 0..self.state.generals.len() {
                if self.state.generals[player] >= 0 {
                    let player_state = self.state.get_player_state(player);
                    let mov = self.players[player].get_move(&player_state, player);

                    // println!("{:?}", mov);
                    moves.push(mov);

                    active_players += 1;
                    last_active = player;
                } else {
                    moves.push(None);
                }
            }

            if !predict && active_players <= 1 {
                return Some(last_active);
            }

            for player in 0..self.state.generals.len() {
                if let Some(mov) = moves[player] {
                    state.do_move(player, mov);
                }
            }

            self.state = state;

            if !predict {
                println!("{}", self.state);
            }

            thread::sleep(wait.saturating_sub(time_spent.elapsed()));
        }

        None
    }
}
