use super::*;

use std::collections::{VecDeque, HashSet};
use std::mem;
use std::time::{Duration, Instant};

use serde_json::Value;
use serde_json::Value as V;

// trait Action: ActionTrait + Player {}

pub trait Action {
    fn len(&self) -> usize;

    fn description(&self) -> String;

    fn generate(&self, smart_bot: &SmartBot, previous: bool) -> Vec<Box<dyn Action>>;

    fn player(&self) -> Box<dyn Player>;
}

pub (super) struct NoneAction{}

impl Action for NoneAction {
    fn len(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        "None".to_string()
    }

    fn generate(&self, _: &SmartBot, _: bool) -> Vec<Box<dyn Action>> {
        vec![Box::new(NoneAction{})]
    }

    fn player(&self) -> Box<dyn Player> {
        Box::new(NoneBot{})
    }
}

struct AdjLand(usize);

impl Action for AdjLand {
    fn len(&self) -> usize {
        self.0
    }

    fn description(&self) -> String {
        format!("Expand onto adjacent land for {}.", self.0)
    }

    fn generate(&self, _: &SmartBot, previous: bool) -> Vec<Box<dyn Action>> {
        let mut out = self.0;

        if previous {
            out = out.saturating_sub(1);
        }
        if out == 0 {
            out = 10;
        }

        vec![Box::new(AdjLand(out))]
    }

    fn player(&self) -> Box<dyn Player> {
        Box::new(FuncBot::new(|state: &State, player: usize| {
            for i in 0..state.armies.len() {
                if state.terrain[i] == player as isize ||
                    (state.turn < 100 && i == state.generals[player] as usize)
                {
                    for n in get_neighbors(state.width, state.height, i) {
                        if state.terrain[n] != player as isize &&
                            state.terrain[n] != TILE_MOUNTAIN &&
                                state.armies[i] > state.armies[n] + 1
                        {
                            return Some(Move::tup((i, n)));
                        }
                    }
                }
            }
            None
        }))
    }
}

struct Gather {
    len: usize,
    loc: usize,
    kind: String,
    start_turn: usize,
    moves: Vec<(usize, usize)>
}

impl Gather {
    fn new(kind: &str) -> Self {
        Self {
            len: 0,
            loc: 0,
            kind: kind.to_string(),
            start_turn: 0,
            moves: Vec::new()
        }
    }
}

impl Action for Gather {
    fn len(&self) -> usize {
        self.len
    }

    fn description(&self) -> String {
        format!("Gather type \"{}\" for {}", self.kind, self.len)
    }

    fn generate(&self, smart_bot: &SmartBot, previous: bool) -> Vec<Box<dyn Action>> {
        let mut out = Vec::new();
        let general = smart_bot.state.generals[smart_bot.player] as usize;

        match &self.kind[..] {
            "general" => {
                for mut len in (1..41).step_by(10) {
                    if previous && self.len > 1 {
                        len = self.len - 1;
                    }

                    out.push(Self{
                        len,
                        loc: general,
                        kind: String::from("general"),
                        start_turn: smart_bot.state.turn,
                        moves: smart_bot.gather(len, general, true, false).0
                    });

                    if previous && self.len > 1 {
                        break;
                    }
                }
            }
            "enemy general" => {
                for (player, general) in smart_bot.state.generals.iter().enumerate() {
                    if player != smart_bot.player && *general >= 0 {
                        let general = *general as usize;
                        let moves = smart_bot.gather_until(1, general, false, true);

                        if moves.len() > 0 {
                            out.push(Self{
                                len: moves.len(),
                                loc: general,
                                kind: String::from("enemy general"),
                                start_turn: smart_bot.state.turn,
                                moves
                            });
                        }
                    }
                }
            }
            "city" => {
                if let Some(city) = smart_bot.find_nearest_city(general) {
                    let moves = smart_bot.gather_until(1, city, false, true);

                    if moves.len() > 0 {
                        out.push(Self{
                            len: moves.len(),
                            loc: city,
                            kind: String::from("city"),
                            start_turn: smart_bot.state.turn,
                            moves
                        });
                    }
                }
            }
            "probe" => {
                for player in 0..smart_bot.state.generals.len() {
                    if player != smart_bot.player {
                        if let Some(loc) = smart_bot.get_probe_loc(player) {
                            let moves = smart_bot.gather_until(1, loc, false, true);

                            if moves.len() > 0 {
                                out.push(Self{
                                    len: moves.len(),
                                    loc,
                                    kind: String::from("probe"),
                                    start_turn: smart_bot.state.turn,
                                    moves
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        out.into_iter().map(|g| Box::new(g) as Box<dyn Action>).collect()
    }

    fn player(&self) -> Box<dyn Player> {
        Box::new(PlayBackBot::new(Move::opts(self.moves.clone()), self.start_turn))
    }
}

struct Expand {
    len: usize,
    loc: usize,
    gather_time: usize,
    start_turn: usize,
    moves: Vec<(usize, usize)>
}

impl Expand {
    fn new() -> Self {
        Self {
            len: 0,
            loc: 0,
            gather_time: 0,
            start_turn: 0,
            moves: Vec::new()
        }
    }
}

impl Action for Expand {
    fn len(&self) -> usize {
        self.len
    }

    fn description(&self) -> String {
        format!("Expand, gathering for {} and total time {}.", self.gather_time, self.len)
    }

    fn generate(&self, smart_bot: &SmartBot, previous: bool) -> Vec<Box<dyn Action>> {
        let (loc, gather_time, iters) =
            if previous && self.len > 1 {
                (Some(self.loc), self.gather_time.saturating_sub(1), 1)
            } else {
                (None, 10, 2)
            };

        let mut out: Vec<Box<dyn Action>> = Vec::new();

        for _ in 0..iters {
            if let Some(loc) = loc.or_else(|| smart_bot.find_strand_loc(5, 5)) {
                let moves1 = smart_bot.gather_expand(loc, gather_time, false);
                let moves2 = smart_bot.gather_expand(loc, gather_time, true);

                let (next_loc1, next_loc2) =
                    if gather_time == 0 {
                        (moves1[0].1, moves2[0].1)
                    } else {
                        (loc, loc)
                    };

                out.push(Box::new(Self {
                    len: moves1.len(),
                    loc: next_loc1,
                    gather_time,
                    start_turn: smart_bot.state.turn,
                    moves: moves1,
                }));

                out.push(Box::new(Self {
                    len: moves2.len(),
                    loc: next_loc2,
                    gather_time,
                    start_turn: smart_bot.state.turn,
                    moves: moves2,
                }));
            }
        }

        out
    }

    fn player(&self) -> Box<dyn Player> {
        Box::new(PlayBackBot::new(Move::opts(self.moves.clone()), self.start_turn))
    }
}

pub(super) fn init_action_gens() -> Vec<Box<dyn Action>> {
    vec![
        Box::new(NoneAction{}) as Box<dyn Action>,
        Box::new(AdjLand(0)),
        Box::new(Gather::new("general")),
        Box::new(Gather::new("enemy general")),
        Box::new(Gather::new("city")),
        Box::new(Gather::new("probe")),
        Box::new(Expand::new()),
    ]
}
