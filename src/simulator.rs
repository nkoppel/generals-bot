use indicatif::ProgressIterator;

use super::state::*;
use std::any::Any;
use std::mem;

fn to_1d(width: usize, x: isize, y: isize) -> isize {
    x + y * width as isize
}

fn from_1d(width: usize, loc: isize) -> (isize, isize) {
    (loc % width as isize, loc / width as isize)
}

pub fn get_neighbor_mask(width: usize, height: usize, loc: usize) -> [bool; 4] {
    let size = width * height;
    [
        loc >= width,
        loc < size && loc / width == (loc + 1) / width,
        loc + width < size,
        loc > 0 && loc / width == (loc - 1) / width,
    ]
}

pub fn get_neighbor_locs(width: usize, _height: usize, loc: usize) -> [usize; 4] {
    [
        loc.saturating_sub(width),
        loc + 1,
        loc + width,
        loc.saturating_sub(1),
    ]
}

pub fn get_neighbors(width: usize, height: usize, loc: usize) -> impl Iterator<Item = usize> {
    let mask = get_neighbor_mask(width, height, loc);
    let locs = get_neighbor_locs(width, height, loc);

    mask.into_iter()
        .zip(locs.into_iter())
        .filter_map(|(valid, loc)| valid.then_some(loc))
}

pub fn select_rand<F: Fn(usize) -> bool>(len: usize, f: F) -> Option<usize> {
    let mut num_true = 0;

    for i in 0..len {
        if f(i) {
            num_true += 1;
        }
    }

    if num_true == 0 {
        return None;
    }

    let mut rng = thread_rng();

    let n = rng.gen_range(0..num_true) + 1;
    let mut j = 0;

    for i in 0..len {
        if f(i) {
            j += 1;
        }
        if j == n {
            return Some(i);
        }
    }

    None
}

pub fn select_rand_eq<T: PartialEq>(vec: &[T], item: &T) -> Option<usize> {
    select_rand(vec.len(), |i| vec[i] == *item)
}

impl State {
    pub fn update_scores(&mut self) {
        self.scores.fill((0, 0));

        for i in 0..self.terrain.len() {
            if let player @ 0.. = self.terrain[i] {
                self.scores[player as usize].0 += 1;
                self.scores[player as usize].1 += self.armies[i] as usize;
            }
        }
    }

    pub fn get_player_state(&self, player: usize) -> Self {
        let deltas = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        let team = self.teams[player];
        let mut out = self.add_fog();

        for loc in 0..self.terrain.len() {
            if self.teams.get(self.terrain[loc] as usize) == Some(&team) {
                let (x, y) = from_1d(self.width, loc as isize);

                for (xi, yi) in &deltas {
                    let x2 = x + xi;
                    let y2 = y + yi;

                    if x2 >= 0 && y2 >= 0 && x2 < self.width as isize && y2 < self.height as isize {
                        let loc2 = to_1d(self.width, x2, y2) as usize;

                        out.terrain[loc2] = self.terrain[loc2];
                        out.armies[loc2] = self.armies[loc2];
                    }
                }
            }
        }

        for (idx, loc) in self.generals.iter().enumerate() {
            if *loc >= 0 {
                if out.terrain[*loc as usize] == TILE_FOG_OBSTACLE {
                    out.terrain[*loc as usize] = TILE_FOG;
                }
                if out.terrain[*loc as usize] > TILE_FOG {
                    out.generals[idx] = *loc;
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
        self.turn += 1;

        if self.turn % 50 == 0 {
            for i in 0..self.terrain.len() {
                if self.terrain[i] >= 0 {
                    self.armies[i] += 1;
                }
            }
        }
        if self.turn % 2 == 0 {
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
        let d = mov.start.abs_diff(mov.end);
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
        ret |= self.armies[mov.start] < 2;

        ret |= self.terrain[mov.end] < -1;

        !ret
    }

    pub fn do_move(&mut self, player: usize, mov: Move) -> bool {
        if !self.move_is_valid(player, mov) {
            return false;
        }

        let transferred = if mov.is50 {
            self.armies[mov.start] / 2
        } else {
            self.armies[mov.start] - 1
        };

        self.armies[mov.start] -= transferred;

        let player2 = self.terrain[mov.end];

        if player2 == player as isize {
            self.armies[mov.end] += transferred;
        } else if self.teams.get(player) == self.teams.get(player2 as usize) {
            self.armies[mov.end] += transferred;

            if self.generals[player2 as usize] != mov.end as isize {
                self.terrain[mov.end] = player as isize;
            }
        } else if transferred > self.armies[mov.end] {
            if player2 >= 0 && self.generals[player2 as usize] == mov.end as isize {
                self.capture_player(player, player2 as usize);
            }

            self.terrain[mov.end] = player as isize;
            self.armies[mov.end] = transferred - self.armies[mov.end];
        } else {
            self.armies[mov.end] -= transferred;
        }

        true
    }

    pub fn get_random_move(&self, player: usize) -> Option<Move> {
        let mut rng = thread_rng();

        let start = select_rand(self.terrain.len(), |i| {
            self.terrain[i] == player as isize && self.armies[i] > 1
        })?;
        let mut neighbors: Vec<usize> = get_neighbors(self.width, self.height, start).collect();

        neighbors.retain(|i| self.terrain[*i] >= -1);

        if neighbors.is_empty() {
            return None;
        }

        let end = neighbors[rng.gen_range(0..neighbors.len())];

        Some(Move::new(start, end, false))
    }

    pub fn step(&mut self, moves: &[Option<Move>]) {
        if self.turn % 2 == 1 {
            for (player, mov) in moves.iter().enumerate().rev() {
                if let Some(mov) = mov {
                    self.do_move(player, *mov);
                }
            }
        } else {
            for (player, mov) in moves.iter().enumerate() {
                if let Some(mov) = mov {
                    self.do_move(player, *mov);
                }
            }
        }

        self.incr_armies();
        self.update_scores();
    }

    // if game is over, return the id of the team that won
    pub fn game_over(&self) -> Option<usize> {
        let mut active_team = None;

        for (player, general) in self.generals.iter().enumerate() {
            if *general > -2 {
                let team = self.teams[player] as usize;

                if active_team.is_some() && active_team != Some(team) {
                    return None;
                }

                active_team = Some(team);
            }
        }

        active_team
    }
}

pub trait Player: Any {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move>;
}

use std::thread;
use std::time::{Duration, Instant};

pub struct Simulator {
    pub state: State,
    pub player_states: Vec<State>,
    players: Vec<Box<dyn Player>>,
}

impl Simulator {
    pub fn new(mut state: State, players: Vec<Box<dyn Player>>) -> Self {
        let player_states = vec![State::new(); state.generals.len()];

        state.remove_fog();
        state.update_scores();

        Self {
            state,
            player_states,
            players,
        }
    }

    pub fn into_state(self) -> State {
        self.state
    }

    pub fn get_players(&mut self) -> Vec<Box<dyn Player>> {
        mem::take(&mut self.players)
    }

    pub fn get_moves(&mut self) -> Vec<Option<Move>> {
        let mut moves = vec![None; self.state.generals.len()];

        if self.player_states[0].turn != self.state.turn || self.state.turn == 0 {
            for (player, state) in self.player_states.iter_mut().enumerate() {
                *state = self.state.get_player_state(player);
            }
        }

        #[allow(clippy::needless_range_loop)]
        for player in 0..self.state.generals.len() {
            moves[player] = self.players[player].get_move(&self.player_states[player], player);
        }

        moves
    }

    pub fn step(&mut self) {
        let moves = self.get_moves();
        self.state.step(&moves);
    }

    pub fn sim(&mut self, rounds: usize, wait: usize, spectate: bool) -> Option<usize> {
        let wait = Duration::from_millis(wait as u64);

        for _ in (0..rounds).progress() {
            let time_spent = Instant::now();

            self.step();

            if let out @ Some(_) = self.state.game_over() {
                return out;
            }

            if spectate {
                println!("{}", self.state);
            }

            thread::sleep(wait.saturating_sub(time_spent.elapsed()));
        }

        None
    }
}
