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
}

impl Player for SmartBot {
    fn init(&mut self, player: usize) {
        self.player = player;
    }

    fn get_move(&mut self, diff: StateDiff) -> Option<Move> {
        self.update(diff);
        println!("{}", self.state);

        let now = Instant::now();
        let out = self.strategy_turtle();

        println!("{}", now.elapsed().as_millis());
        println!("{:?}", self.modes);
        println!("{:?}", self.moves);

        out
    }
}
