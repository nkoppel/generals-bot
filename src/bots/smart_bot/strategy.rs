use super::*;

const PROBE_SURPLUS: usize = 10;

pub(super) struct Action {
    description: String,
    len: usize,
    player: Box<dyn Player>
}

impl Action {
    pub fn empty() -> Self {
        Self {
            description: String::new(),
            len: 0,
            player: Box::new(NoneBot{})
        }
    }
}

impl SmartBot {
    fn get_actions(&mut self) -> Vec<Action> {
        let mut out = Vec::new();
        let mut moves;
        let capital = self.state.generals[self.player] as usize;

        out.push(mem::replace(&mut self.previous_action, Action::empty()));

        // Gather on capital
        for len in [10, 20, 40] {
            out.push(
                Action {
                    description: format!("Gather on capital for {}", len),
                    len,
                    player: Box::new(PlayBackBot::new(
                        Move::opts(self.gather(len, capital, true, false).0),
                        self.state.turn
                    ))
                }
            );
        }

        // Attempt to capture enemy capitals
        for player in 0..self.state.generals.len() {
            if player == self.player {
                continue;
            }

            if self.seen_generals[player] >= 0 {
                moves = self.gather_until(1, self.seen_generals[player] as usize, false, true);

                out.push(
                    Action {
                        description: format!("Gather on player {}'s capital", player),
                        len: moves.len(),
                        player: Box::new(PlayBackBot::new(Move::opts(moves), self.state.turn))
                    }
                );
            }
        }

        // Capture a city
        if let Some(city) = self.find_nearest_city(capital) {
            moves = self.gather_until(1, city, false, true);

            out.push(
                Action {
                    description: format!("Capture city at {}", city),
                    len: moves.len(),
                    player: Box::new(PlayBackBot::new(Move::opts(moves), self.state.turn))
                }
            );
        }

        // Probe enemy territory
        for player in 0..self.state.generals.len() {
            if let Some(probe_loc) = self.get_probe_loc(player) {
                moves = self.gather_until(PROBE_SURPLUS, probe_loc, false, true);

                out.push(
                    Action {
                        description: format!("Probe to {}", probe_loc),
                        len: moves.len(),
                        player: Box::new(PlayBackBot::new(Move::opts(moves), self.state.turn))
                    }
                );
            }
        }

        println!("general: {}", self.state.generals[self.player]);
        // Expand, maximizing surface area

        if let Some(loc) = self.find_strand_loc(5, 5) {
            moves = self.gather_expand(loc, 10, false);

            out.push(
                Action {
                    description: format!("Expand to {}, maximizing surface area", moves.last().unwrap().1),
                    len: moves.len(),
                    player: Box::new(PlayBackBot::new(Move::opts(moves), self.state.turn))
                }
            );
        }

        // Expand, minimizing surface area
        if let Some(loc) = self.find_strand_loc(5, 5) {
            moves = self.gather_expand(loc, 10, true);

            out.push(
                Action {
                    description: format!("Expand to {}, minimizing surface area", moves.last().unwrap().1),
                    len: moves.len(),
                    player: Box::new(PlayBackBot::new(Move::opts(moves), self.state.turn))
                }
            );
        }

        // Expand to adjacent land
        // out.push(
            // Action {
                // description: format!("Expand to adjacent land"),
                // len: 10,
                // player: Box::new(FuncBot::new(Box::new(|s, p| Move::opt(adj_land(s, p)))))
            // }
        // );

        out
    }

    fn get_action_move(&self, action: &mut Action) -> Option<Move> {
        action.player.init(self.player);
        action.player.get_move(State::new().diff(&self.state))
    }

    fn score_action(&self, action: &mut Action) -> f64 {
        let mut players: Vec<Box<dyn Player>> = Vec::new();

        for i in 0..self.state.generals.len() {
            players.push(Box::new(NoneBot{}));
        }

        players[self.player] = mem::replace(&mut action.player, Box::new(NoneBot{}));

        let mut simulator = Simulator::new(self.state.clone(), players);
        simulator.sim(action.len, 0, true);

        players = simulator.get_players();
        action.player = mem::replace(&mut players[self.player], Box::new(NoneBot{}));

        let state = simulator.into_state().get_player_state(self.player);

        // if action.description.len() > 9 && action.description[0..9] == *"Expand to" {
            // println!("{}", state)
        // }

        let mut armies = state.scores[self.player].1 as isize - state.scores[self.player].0 as isize;

        armies += 2 * state.armies[state.generals[self.player] as usize];

        // Rate of army production per 25 second
        let mut production = 24;

        for city in &state.cities {
            if state.terrain[*city as usize] == self.player as isize {
                production += 24;
            }
        }

        production += state.scores[self.player].0;

        // Total seen land
        let mut seen = 0;

        for loc in 0..state.terrain.len() {
            if state.terrain[loc] >= -2 || self.seen_terrain[loc] >= 0 {
                seen += 1
            }
        }

        let mut out = 0.;

        out += 4. * (0.5_f64.powf(1. / 25.)).powi(production as i32);
        out += 1. * 0.8_f64.powi(armies as i32);
        out += 32. * 0.95_f64.powi(seen as i32);

        for i in 0..state.generals.len() {
            if i == self.player {
                continue;
            }
            if state.terrain.get(self.seen_generals[i] as usize) == Some(&(self.player as isize)) {
                out /= 10.;
            }
        }

        println!("{} {} {}", production, armies, seen);

        out
    }
}

impl Player for SmartBot {
    fn init(&mut self, player: usize) {
        *self = Self::new();
        self.player = player;
    }

    fn get_move(&mut self, diff: StateDiff) -> Option<Move> {
        self.update(diff);
        // println!("{}", self.state);

        let now = Instant::now();
        let base_score = self.score_action(&mut Action::empty());
        let mut actions = self.get_actions();
        let mut best_action = Action::empty();
        let mut best_efficiency = f64::NEG_INFINITY;

        for (i, mut action) in actions.into_iter().enumerate() {
            let score = self.score_action(&mut action);
            let mut efficiency = (base_score - score) / action.len as f64;

            if action.len == 0 {
                efficiency = 0.;
            }
            if i == 0 {
                efficiency *= 10.;
            }

            println!("{:?}: {} {}", action.description, score, efficiency);

            if efficiency > best_efficiency {
                best_action = action;
                best_efficiency = efficiency;
            }
        }

        println!("{}", best_action.description);
        println!("{}", now.elapsed().as_millis());

        let out = self.get_action_move(&mut best_action);
        if best_action.len >= 1 {
            best_action.len -= 1;
            self.previous_action = best_action;
        }
        out
    }
}
