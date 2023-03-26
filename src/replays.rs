use crate::simulator::*;
use crate::state::*;
use crate::PlayBackBot;

use std::io::Read;

use anyhow::Result;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct ReplayMove {
    index: usize,
    start: isize,
    end: isize,
    is50: usize,
    turn: usize,
}

// TODO: if parsing FFA matches for replay, we need to implement players' land
// turning grey due to surrender
#[derive(Clone, Debug, Deserialize)]
struct ReplayAFK {
    index: usize,
    turn: usize,
}

#[allow(non_snake_case)]
#[derive(Clone, Debug, Deserialize)]
pub struct Replay {
    id: String,
    mapWidth: usize,
    mapHeight: usize,
    usernames: Vec<String>,
    cities: Vec<isize>,
    cityArmies: Vec<isize>,
    generals: Vec<isize>,
    mountains: Vec<usize>,
    moves: Vec<ReplayMove>,
    afks: Vec<ReplayAFK>,
    teams: Option<Vec<isize>>,
}

impl Replay {
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        Ok(serde_json::de::from_reader(reader)?)
    }

    pub fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::de::from_str(s)?)
    }

    pub fn read_from_file(path: &str) -> Result<Self> {
        Self::from_reader(std::fs::File::open(path)?)
    }

    pub fn num_players(&self) -> usize {
        self.generals.len()
    }

    pub fn to_simulator(&self) -> (Simulator, usize) {
        let mut moves = vec![Vec::new(); self.generals.len()];

        for mov in &self.moves {
            let turn = mov.turn - 1;

            if turn + 1 > moves[0].len() {
                for l in &mut moves {
                    l.resize(turn + 1, None);
                }
            }

            moves[mov.index][turn] = Some(Move {
                start: mov.start as usize,
                end: mov.end as usize,
                is50: mov.is50 != 0,
            });
        }

        let turns = moves[0].len();

        let bots = moves
            .into_iter()
            .map(|moves| {
                Box::new(PlayBackBot {
                    moves,
                    start_turn: 1,
                }) as Box<dyn Player>
            })
            .collect::<Vec<_>>();

        let size = self.mapWidth * self.mapHeight;
        let mut state = State {
            width: self.mapWidth,
            height: self.mapHeight,
            turn: 0,
            terrain: vec![TILE_EMPTY; size],
            armies: vec![0; size],
            cities: self.cities.clone(),
            generals: self.generals.clone(),
            scores: vec![(0, 0); self.generals.len()],
            teams: self
                .teams
                .clone()
                .unwrap_or_else(|| (0..self.generals.len() as isize).collect()),
        };

        for i in 0..self.cities.len() {
            state.armies[self.cities[i] as usize] = self.cityArmies[i];
        }

        for i in 0..self.generals.len() {
            state.terrain[self.generals[i] as usize] = i as isize;
            state.armies[self.generals[i] as usize] = 1;
        }

        for i in &self.mountains {
            state.terrain[*i] = TILE_MOUNTAIN;
        }

        (Simulator::new(state, bots), turns)
    }
}
