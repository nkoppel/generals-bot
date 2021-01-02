use super::state::*;

fn to_1d(width: usize, x: isize, y: isize) -> isize {
    x + y * width as isize
}

fn from_1d(width: usize, loc: isize) -> (isize, isize) {
    (loc % width as isize, loc / width as isize)
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
        let mut out = self.copy_fog();

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
        }

        self.turn += 1;
    }

    fn capture_player(&mut self, p1: usize, p2: usize) {
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
                self.armies[mov.end] -= self.armies[mov.start] - 1;
            }
        }

        true
    }

    pub fn get_random_move(&mut self, player: usize) -> Option<Move> {
        let player_tiles = self.scores[player].0;

        let mut rng = thread_rng();

        let tile = rng.gen_range(0, player_tiles);
        let mut start = 0;
        let mut j = 0;

        while start < self.terrain.len() {
            if self.terrain[start] == player as isize {
                if j == tile {
                    break;
                }

                j += 1;
            }

            start += 1;
        }

        let ds = vec![-1, 1, -(self.width as isize), self.width as isize]
            .into_iter()
            .filter(|d| {
                    let end = (start as isize + d) as usize;

                    self.move_is_valid(player, Move::new(start, end, false)) 
            })
            .collect::<Vec<_>>();

        if ds.is_empty() {
            return None;
        }

        let d = ds[rng.gen_range(0, ds.len())];
        let end = (start as isize + d) as usize;

        Some(Move::new(start, end, false))
    }
}

pub trait Player {
    fn init(&mut self, teams: &Vec<usize>, player: usize);
    fn get_move(&mut self, diff: StateDiff) -> Option<Move>;
}

pub struct Simulator {
    state: State,
    player_states: Vec<State>,
    players: Vec<Box<dyn Player>>,
}

impl Simulator {
    pub fn new(state: State, mut players: Vec<Box<dyn Player>>) -> Self {
        let player_states = vec![State::new(); state.generals.len()];
        let teams = (1..players.len() + 1).collect::<Vec<_>>();

        for i in 0..players.len() {
            players[i].init(&teams, i);
        }

        Self {
            state,
            player_states,
            players,
        }
    }

    pub fn sim(&mut self, rounds: usize) -> Option<usize> {
        // println!("{}", self.state);

        for _ in 0..rounds {
            self.state.incr_armies();
            self.state.update_scores();

            let mut state = self.state.clone();
            let mut moves = Vec::new();
            let mut active_players = 0;
            let mut last_active = 0;

            for player in 0..self.state.generals.len() {
                if self.state.generals[player] >= 0 {
                    let player_state = self.state.get_player_state(player);
                    let diff = self.player_states[player].diff(&player_state);

                    let mov = self.players[player].get_move(diff);

                    // println!("{:?}", mov);
                    moves.push(mov);

                    if let Some(m) = mov {
                        debug_assert!(state.move_is_valid(player, m));
                    }

                    self.player_states[player] = player_state;
                    active_players += 1;
                    last_active = player;
                } else {
                    moves.push(None);
                }
            }

            for player in 0..self.state.generals.len() {
                if let Some(mov) = moves[player] {
                    state.do_move(player, mov);
                }
            }

            if active_players <= 1 {
                return Some(last_active);
            }

            self.state = state;

            // println!("{}", self.state);
        }

        None
    }
}
