pub const TILE_EMPTY       : isize = -1;
pub const TILE_MOUNTAIN    : isize = -2;
pub const TILE_FOG         : isize = -3;
pub const TILE_FOG_OBSTACLE: isize = -4;

pub use rand::thread_rng;
pub use rand::random;
pub use rand::Rng;
pub use rand::seq::SliceRandom;

fn diff(old: &Vec<isize>, new: &Vec<isize>) -> Vec<isize> {
    let mut out = Vec::new();
    let mut start = true;
    let mut i = 0;
    let mut j = 0;

    while (i < old.len() && i < new.len()) || start {
        start = false;

        while i < old.len() && i < new.len() && old[i] == new[i] {
            j += 1;
            i += 1;
        }

        out.push(j);
        out.push(0);
        j = 0;

        while i < new.len() && (i >= old.len() || old[i] != new[i]) {
            out.push(new[i]);
            j += 1;
            i += 1;
        }

        let len = out.len();

        out[len - 1 - j as usize] = j;
        j = 0;
    }

    out
}

fn patch(vec: &Vec<isize>, diff: &Vec<isize>) -> Vec<isize> {
    let mut out = Vec::new();
    let mut i = 0;

    while i < diff.len() {
        if diff[i] != 0 {  // matching
            out.extend_from_slice(&vec[out.len()..out.len() + diff[i] as usize]);
        }
        i += 1;

        if i < diff.len() && diff[i] != 0 {  // mismatching
            out.extend_from_slice(&diff[i + 1..i + 1 + diff[i] as usize]);
            i += diff[i] as usize;
        }
        i += 1;
    }

    return out;
}

#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub start: usize,
    pub end: usize,
    pub is50: bool
}

impl Move {
    pub fn new(start: usize, end: usize, is50: bool) -> Self {
        Move{start, end, is50}
    }

    pub fn tup((start, end): (usize, usize)) -> Self {
        Move{start, end, is50: false}
    }

    pub fn opt(mov: Option<(usize, usize)>) -> Option<Self> {
        match mov {
            None => None,
            Some(m) => Some(Self::tup(m))
        }
    }

    pub fn tups(vec: Vec<(usize, usize)>) -> Vec<Self> {
        vec.into_iter().map(Move::tup).collect()
    }

    pub fn opts(vec: Vec<(usize, usize)>) -> Vec<Option<Self>> {
        vec.into_iter().map(|m| Some(Move::tup(m))).collect()
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub width: usize,
    pub height: usize,
    pub turn: usize,
    pub terrain: Vec<isize>,
    pub armies: Vec<isize>,
    pub cities: Vec<isize>,
    pub generals: Vec<isize>,
    pub scores: Vec<(usize, usize)>,
    pub teams: Vec<isize>,
}

#[derive(Clone, Debug)]
pub struct StateDiff {
    pub turn: usize,
    pub map_diff: Vec<isize>,
    pub cities_diff: Vec<isize>,
    pub generals: Vec<isize>,
    pub scores: Vec<(usize, usize)>
}

impl State {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            turn: 0,
            terrain: Vec::new(),
            armies: Vec::new(),
            cities: Vec::new(),
            generals: Vec::new(),
            scores: Vec::new(),
            teams: Vec::new(),
        }
    }

    pub fn with_teams(teams: Vec<isize>) -> Self {
        Self {
            teams,
            .. Self::new()
        }
    }

    pub fn add_fog(&self) -> Self {
        let mut out = self.clone();
        let size = self.size();

        out.cities = Vec::new();
        out.armies = vec![0; size];
        out.generals = vec![-1; self.generals.len()];

        out.terrain = vec![TILE_FOG; size];

        for i in 0..self.terrain.len() {
            if self.terrain[i] == TILE_MOUNTAIN {
                out.terrain[i] = TILE_FOG_OBSTACLE;
            }
        }

        for i in &self.cities {
            out.terrain[*i as usize] = TILE_FOG_OBSTACLE;
        }

        out
    }

    pub fn remove_fog(&mut self) {
        for i in 0..self.terrain.len() {
            self.terrain[i] =  match self.terrain[i] {
                -3 => -1,
                -4 => -2,
                n => n
            }
        }
    }

    pub fn size(&self) -> usize {
        self.width * self.height
    }

    pub fn generate(width: usize, height: usize, nmountians: usize, ncities: usize, nplayers: usize) -> Self {
        let size = width * height;

        if nmountians + ncities + nplayers >= size {
            panic!("Map is too small for {} mountians, {} cities, and {} players", nmountians, ncities, nplayers);
        }

        let mut rng = thread_rng();

        let mut tiles: Vec<usize> = (0..size).collect();
        let mut i = 0;

        tiles.shuffle(&mut rng);

        let mut out = Self {
            width,
            height,
            turn: 0,
            terrain: vec![TILE_EMPTY; size],
            armies: vec![0; size],
            cities: Vec::new(),
            generals: vec![0; nplayers],
            scores: vec![(0, 0); nplayers],
            teams: (0..nplayers as isize).collect::<Vec<_>>(),
        };

        for _ in 0..nmountians {
            out.terrain[tiles[i]] = TILE_MOUNTAIN;
            i += 1;
        }

        for _ in 0..ncities {
            out.cities.push(tiles[i] as isize);
            out.armies[tiles[i]] = rng.gen_range(40..51);
            i += 1;
        }

        for j in 0..nplayers {
            out.generals[j] = tiles[i] as isize;
            out.terrain[tiles[i]] = j as isize;
            out.armies[tiles[i]] = 1;
            i += 1;
        }

        out
    }

    fn serialize_map(&self) -> Vec<isize> {
        let mut out = vec![self.width as isize, self.height as isize];
        out.extend_from_slice(&self.armies[..]);
        out.extend_from_slice(&self.terrain[..]);

        out
    }

    fn deserialize_map(&mut self, data: &Vec<isize>) {
        self.width  = data[0] as usize;
        self.height = data[1] as usize;

        let size = self.size();

        self.armies = data[2..2 + size].to_vec();
        self.terrain = data[2 + size..].to_vec();
    }

    pub fn diff(&self, new: &Self) -> StateDiff {
        let map1 = self.serialize_map();
        let map2 = new.serialize_map();

        StateDiff {
            turn: new.turn,
            map_diff: diff(&map1, &map2),
            cities_diff: diff(&self.cities, &new.cities),
            generals: new.generals.clone(),
            scores: new.scores.clone()
        }
    }

    pub fn patch(&mut self, diff: StateDiff) {
        let map = self.serialize_map();

        self.turn = diff.turn;
        self.deserialize_map(&patch(&map, &diff.map_diff));
        self.cities = patch(&self.cities, &diff.cities_diff);
        self.generals = diff.generals;
        self.scores = diff.scores;

        if self.teams.is_empty() {
            self.teams = (0..self.generals.len() as isize).collect();
        }
    }
}

pub fn display_turn(turn: usize) -> String {
    if turn % 2 == 0 {
        format!("{} ", turn / 2)
    } else {
        format!("{}.", turn / 2)
    }
}

fn pad_front(c: char, len: usize, s: &str) -> String {
    let mut out = String::new();

    if s.len() > len {
        return s.to_string();
    }

    for _ in 0..len - s.len() {
        out.push(c);
    }

    out += s;

    out
}

use std::fmt;

use colored::*;
use colored::Color::*;

fn show_num(n: usize) -> String {
    match n {
        0                       => "".to_string(),
        1..=9_999               => format!("{}", n),
        10_000..=999_999        => format!("{}K", n / 1000),
        1_000_000..=999_999_999 => format!("{}M", n / 1_000_000),
        _                       => format!("{}B", n / 1_000_000_000),
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let player_colors = vec![Red, Blue, Green, Cyan, Magenta];

        writeln!(f, "{}", display_turn(self.turn))?;

        for y in 0..self.height {
            for x in 0..self.width {
                let i = x + y * self.width;

                let mut back = Black;
                let mut front = White;
                let pad;

                let tmp = self.generals
                    .clone()
                    .iter_mut()
                    .position(|x| *x == i as isize);

                if let Some(n) = tmp {
                    pad = '\\';
                    front = player_colors[n as usize]
                } else if self.cities.contains(&(i as isize)) {
                    pad = '0';

                    if self.terrain[i] >= 0 {
                        front = player_colors[self.terrain[i] as usize];
                    }
                } else {
                    match self.terrain[i] {
                        TILE_EMPTY => pad = '.',
                        TILE_MOUNTAIN => pad = '^',
                        TILE_FOG => {pad = '.'; back = BrightBlack},
                        TILE_FOG_OBSTACLE => {pad = '^'; back = BrightBlack},
                        n => {
                            pad = '.';
                            front = player_colors[n as usize]
                        }
                    }
                }

                let num = show_num(self.armies[i] as usize);

                write!(f, "{} ", pad_front(pad, 4, &num).on_color(back).color(front))?;
            }
            writeln!(f)?;
        }

        writeln!(f)
    }
}
