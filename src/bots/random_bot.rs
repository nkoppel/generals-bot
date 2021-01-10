use super::*;

pub struct RandomBot {
    state: State,
    player: usize
}

impl RandomBot {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            player: 0
        }
    }
}

impl Player for RandomBot {
    fn init(&mut self, _: &Vec<usize>, player: usize) {
        self.player = player;
    }

    fn get_move(&mut self, diff: StateDiff) -> Option<Move> {
        self.state.patch(diff);
        // println!("{}", self.state);
        self.state.get_random_move(self.player)
    }
}

pub struct NoneBot{}

impl Player for NoneBot {
    fn init(&mut self, _: &Vec<usize>, _: usize) {}

    fn get_move(&mut self, _: StateDiff) -> Option<Move> {
        None
    }
}
