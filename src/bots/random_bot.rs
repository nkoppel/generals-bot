use super::*;

pub struct RandomBot {
    state: State,
    player: usize
}

impl RandomBot {
    pub fn new(player: usize) -> Self {
        Self {
            state: State::new(),
            player
        }
    }
}

impl Player for RandomBot {
    fn get_move(&mut self, diff: StateDiff) -> Option<Move> {
        self.state.patch(diff);

        // if self.player == 0 {
            // println!("{}", self.state);
        // }

        self.state.get_random_move(self.player)
    }
}
