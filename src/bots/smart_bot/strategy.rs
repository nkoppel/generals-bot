use super::*;

impl SmartBot {
    fn score_state(&self, state: &State) -> f64 {
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

        // println!("{} {} {}", production, armies, seen);

        out
    }

    fn score_action(&self, action: &Box<dyn Action>) -> f64 {
        let mut players: Vec<Box<dyn Player>> = Vec::new();

        for i in 0..self.state.generals.len() {
            players.push(Box::new(NoneBot{}));
        }

        players[self.player] = action.player();

        let mut simulator = Simulator::new(self.state.clone(), players);
        simulator.sim(action.len(), 0, true);

        let state = simulator.into_state().get_player_state(self.player);

        self.score_state(&state)
    }
}

impl Player for SmartBot {
    fn reset(&mut self) {
        *self = Self::new();
    }

    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        self.player = player;
        self.update(state);

        let now = Instant::now();

        let mut actions = Vec::new();

        for i in 0..self.action_gens.len() {
            if i == self.previous_type {
                actions.push(self.previous_action.generate(&self, true));
            } else {
                actions.push(self.action_gens[i].generate(&self, true));
            }
        }

        let base_score = self.score_state(&self.state);

        let mut best_efficiency = f64::NEG_INFINITY;
        let mut best_action: Box<dyn Action> = Box::new(NoneAction{});
        let mut best_type = 0;

        for (t, acts) in actions.into_iter().enumerate() {
            for act in acts {
                let score = self.score_action(&act);
                let efficiency = (base_score - score) / act.len() as f64;

                if efficiency > best_efficiency {
                    best_efficiency = efficiency;
                    best_action = act;
                    best_type = t;
                }
            }
        }

        let mov = best_action
            .player()
            .get_move(&self.state, self.player);

        println!("{}", best_action.description());
        println!("{}", now.elapsed().as_millis());

        self.previous_action = best_action;
        self.previous_type   = best_type;

        mov
    }
}
