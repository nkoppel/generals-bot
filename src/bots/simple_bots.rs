use super::*;

pub struct RandomBot {}

impl Player for RandomBot {
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        // println!("{}", state);
        state.get_random_move(player)
    }
}

pub struct NoneBot {}

impl Player for NoneBot {
    fn get_move(&mut self, _: &State, _: usize) -> Option<Move> {
        None
    }
}

pub struct PlayBackBot {
    pub moves: Vec<Option<Move>>,
    pub start_turn: usize,
}

impl PlayBackBot {
    pub fn new(moves: Vec<Option<Move>>, start_turn: usize) -> Self {
        Self { moves, start_turn }
    }
}

impl Player for PlayBackBot {
    fn get_move(&mut self, state: &State, _: usize) -> Option<Move> {
        self.moves
            .get(state.turn.wrapping_sub(self.start_turn))
            .cloned()
            .unwrap_or(None)
    }
}

pub struct FuncBot<T> {
    func: T,
}

impl<T> FuncBot<T>
where
    T: FnMut(&State, usize) -> Option<Move>,
{
    pub fn new(func: T) -> Self {
        Self { func }
    }
}

impl<T> Player for FuncBot<T>
where
    T: FnMut(&State, usize) -> Option<Move>,
{
    fn get_move(&mut self, state: &State, player: usize) -> Option<Move> {
        (self.func)(state, player)
    }
}
