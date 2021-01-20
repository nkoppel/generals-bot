use super::*;

impl SmartBot {
    fn strategy_turtle(&mut self) -> Option<Move> {
        let capital = self.state.generals[self.player] as usize;
        let time = 50 - self.state.turn % 50;

        if self.modes.is_empty() && self.moves.is_empty() {
            if self.state.turn < 5 {
                self.opener(capital);
            } else {
                if let Some(mov) = self.adj_land() {
                    self.moves.push_back(mov);
                } else {
                    self.modes.push_back(Gather_for(time, capital, true, false));
                }
            }
        }
        self.run_modes()
    }

    fn strategy_find_destroy(&mut self) -> Option<Move> {
        let capital = self.state.generals[self.player] as usize;
        let time = 51 - self.state.turn % 50;

        if self.state.turn <= 1 && self.moves.is_empty() && self.modes.is_empty() {
            self.opener(capital);
        }

        if let Some(player) = self.seen_terrain
            .iter()
                .filter(|t| **t >= 0 && **t != self.player as isize)
                .next()
        {
            let player = *player as usize;
            let mut general = usize::MAX;

            for (i, g) in self.seen_generals.iter().enumerate() {
                if i != self.player && *g >= 0 {
                    general = *g as usize;
                }
            }

            if general < usize::MAX {
                if let Some(Gather_until(..)) = self.modes.front() {
                } else {
                    self.modes.push_front(
                        Gather_until(5, general, false, false)
                    );

                    if let Some(mov) = self.run_modes() {
                        return Some(mov);
                    }
                }
            } else if self.modes.is_empty() {
                if let Some(loc) = self.find_probe_loc(player) {
                    self.modes.push_back(Gather_until(self.state.scores[player].1 / 2, loc, false, false));

                    self.modes.push_back(Probe_from(loc, player));

                    if let Some(mov) = self.run_modes() {
                        return Some(mov);
                    }
                }
            }
        }

        if let Some(mov) = self.run_modes() {
            Some(mov)
        } else {
            if self.losing_on_cities() {
                self.get_city();
                self.run_modes() 
            } else if let Some(mov) = self.adj_land() {
                Some(Move::tup(mov))
            } else if self.modes.is_empty() {
                self.modes.push_front(Gather_for(time, capital, true, false));
                self.run_modes()
            } else {
                None
            }
        }
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
        let out = self.strategy_find_destroy();

        // println!("{}", now.elapsed().as_millis());
        // println!("{:?}", self.modes);
        // println!("{:?}", self.moves);

        out
    }
}
