use super::simulator::Player;
use super::state::*;

use std::net::TcpStream;
use std::{thread, time::Duration};

use tungstenite::stream::MaybeTlsStream;
use tungstenite::Error::*;
use tungstenite::Message::*;
use tungstenite::*;

use serde::Deserialize;
use serde_json::{json, Value};

use chrono::prelude::*;

const SECOND: Duration = Duration::from_secs(1);

#[macro_export]
macro_rules! log {
    ($type:literal; $( $format_args:tt ),+) => {
        println!("{} {{ {} }} {}", Local::now().format("[ %T %D ]"), $type, format!($( $format_args ),+));
    };
    ($( $format_args:tt ),+) => {
        println!("{} {}", Local::now().format("[ %T %D ]"), format!($( $format_args ),+));
    }
}

const SLEEP_TIME: u64 = 300;

#[derive(Debug, Clone, PartialEq)]
pub enum GameErr {
    Won,
    Lost,
    ConnectionLost,
}

pub struct Client {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    url: String,
    id: String,
}

impl Client {
    pub fn new(url: &str, i: &str) -> Self {
        Client {
            socket: connect(url).expect("Unable to connect").0,
            url: url.to_string(),
            id: i.to_string(),
        }
    }

    fn send_message(&mut self, value: Value) {
        let mut buf = "42".to_string();
        buf += &value.to_string();

        self.socket.write_message(Text(buf)).unwrap();
        self.socket.write_pending().unwrap();
    }

    pub fn ping(&mut self) {
        self.socket.write_message(Ping(Vec::new())).unwrap();
    }

    pub fn set_username(&mut self, name: &str) {
        self.send_message(json!(["set_username", &self.id, name]));
    }

    pub fn join_private(&mut self, game_id: &str) {
        self.send_message(json!(["join_private", game_id, &self.id]));
    }

    pub fn join_1v1(&mut self) {
        self.send_message(json!(["join_1v1", &self.id]));
    }

    pub fn join_ffa(&mut self) {
        self.send_message(json!(["play", &self.id]));
    }

    pub fn set_force_start(&mut self, game_id: &str) {
        let msg = json!(["set_force_start", game_id, true]);

        self.ping();
        self.send_message(msg.clone());

        loop {
            match self.socket.read_message() {
                Ok(Pong(_)) => {
                    thread::sleep(SECOND);
                    self.ping();
                    self.send_message(msg.clone());
                }
                Err(_) | Ok(Close(_)) => break,
                Ok(Text(s)) => {
                    if let Ok(Value::Array(val)) = serde_json::from_str(&s[2..]) {
                        if val[0] == Value::String("queue_update".to_string()) {
                            if let Some(map) = val[1].as_object() {
                                if let Some(Value::Bool(true)) = map.get("isForcing") {
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn debug_listen(&mut self) {
        self.ping();

        loop {
            match self.socket.read_message() {
                Ok(Pong(_)) => {
                    thread::sleep(SECOND);
                    self.ping()
                }
                Err(_) | Ok(Close(_)) => break,
                Ok(Text(s)) => {
                    if let Ok(val) = serde_json::from_str::<Value>(&s[2..]) {
                        log!("{}", val);
                    } else {
                        log!("Ok({:?})", s);
                    }
                }
                msg => {
                    log!("{:?}", msg);
                }
            }
        }
    }

    pub fn send_move(&mut self, mov: Option<Move>) {
        let mov = mov.unwrap_or(Move::new(0, 0, false));

        self.send_message(json!(["attack", mov.start, mov.end, mov.is50]));
    }

    pub fn get_message<F, T>(&mut self, func: F, default: T) -> T
    where
        F: Fn(Value) -> Option<T>,
    {
        loop {
            match self.socket.read_message() {
                Err(_) | Ok(Close(_)) => return default,
                Ok(Text(msg)) => {
                    if let Ok(val) = serde_json::from_str(&msg[2..]) {
                        if let Some(out) = func(val) {
                            return out;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn get_diff(&mut self) -> Result<StateDiff, GameErr> {
        let f = |val| {
            if let Value::Array(arr) = val {
                if let Value::String(c) = &arr[0] {
                    if c == "game_won" {
                        return Some(Err(GameErr::Won));
                    } else if c == "game_lost" {
                        return Some(Err(GameErr::Lost));
                    } else if c == "game_update" {
                        return Some(Ok(diff_from_value(&arr[1])));
                    }
                }
            }
            None
        };

        self.get_message(f, Err(GameErr::ConnectionLost))
    }

    pub fn get_game_start(&mut self) -> Option<GameStart> {
        let f = |val| {
            if let Value::Array(arr) = val {
                if let Value::String(s) = &arr[0] {
                    if s == "game_start" {
                        return serde_json::from_str(&serde_json::to_string(&arr[1]).unwrap())
                            .unwrap();
                    }
                }
            }
            None
        };

        self.get_message(f, None)
    }

    pub fn run_player(&mut self, player: &mut Box<dyn Player>) -> GameErr {
        let start;

        if let Some(s) = self.get_game_start() {
            start = s;
        } else {
            return GameErr::ConnectionLost;
        }

        log!("Join match"; "type: {}", (start.game_type));
        log!("Join match"; "players: {}", (start.usernames.join(" | ")));
        log!("Join match"; "replay url: http://bot.generals.io/replays/{}", (start.replay_id));

        let player_index = start.playerIndex;
        let mut state = State::new();

        loop {
            let msg = self.get_diff();

            if let Ok(diff) = msg {
                state.patch(diff);

                println!("{}", state);

                self.send_move(player.get_move(&state, player_index));
            } else {
                let err = msg.unwrap_err();
                log!("Leave match"; "result: {:?}", err);
                return err;
            }
        }
    }

    pub fn run_1v1(&mut self, player: &mut Box<dyn Player>) {
        loop {
            log!("Join 1v1"; "");
            self.join_1v1();

            if self.run_player(player) == GameErr::ConnectionLost {
                self.socket = connect(&self.url).expect("Unable to connect").0;
                // } else {
                // thread::sleep(Duration::from_secs(SLEEP_TIME));
            }
        }
    }
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct GameUpdate {
    scores: Vec<HashMap<String, Value>>,
    turn: usize,
    attackIndex: usize,
    generals: Vec<isize>,
    map_diff: Vec<isize>,
    cities_diff: Vec<isize>,
}

use std::collections::HashMap;
use std::result::Result;

fn diff_from_value(val: &Value) -> StateDiff {
    let update: GameUpdate = serde_json::from_str(&serde_json::to_string(val).unwrap()).unwrap();

    let as_usize = |o: Option<&Value>| o.unwrap().as_i64().unwrap() as usize;

    let scores = update
        .scores
        .iter()
        .map(|m| {
            (
                as_usize(m.get("i")),
                as_usize(m.get("tiles")),
                as_usize(m.get("total")),
            )
        })
        .collect::<Vec<_>>();

    let mut scores2 = vec![(0, 0); scores.len()];

    for (i, land, armies) in scores {
        scores2[i] = (land, armies);
    }

    StateDiff {
        turn: update.turn,
        map_diff: update.map_diff,
        cities_diff: update.cities_diff,
        generals: update.generals,
        scores: scores2,
    }
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
pub struct GameStart {
    pub chat_room: String,
    pub game_type: String,
    pub playerIndex: usize,
    pub replay_id: String,
    pub swamps: Vec<isize>,
    pub usernames: Vec<String>,
    pub teams: Vec<isize>,
}

impl Drop for Client {
    fn drop(&mut self) {
        self.socket.close(None).unwrap();

        let mut tmp = self.socket.read_message();

        for _ in 0..1000 {
            if let Err(ConnectionClosed) = tmp {
                break;
            }

            if let Err(AlreadyClosed) = tmp {
                break;
            }

            tmp = self.socket.read_message();
        }
    }
}
