use super::state::*;
use super::simulator::Player;

use std::{thread, time::Duration};
use std::net::TcpStream;

use tungstenite::*;
use tungstenite::Message::*;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::Error::*;

use serde::Deserialize;
use serde_json::{json, Value};

use chrono::prelude::*;

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
    GameWon,
    GameLost,
    GameConnectionLost
}

pub use GameErr::*;

pub struct Client {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    url: String,
    id: String
}

impl Client {
    pub fn new(url: &str, i: &str) -> Self {
        Client {
            socket: connect(url).expect("Unable to connect").0,
            url: url.to_string(),
            id: i.to_string()
        }
    }

    fn send_message(&mut self, value: Value) {
        let mut buf = "42".to_string();
        buf += &value.to_string();

        self.socket.write_message(Text(buf));
        self.socket.write_pending();
    }

    pub fn ping(&mut self) {
        self.socket.write_message(Ping(Vec::new()));
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
        let second = Duration::from_secs(1);

        self.ping();
        self.send_message(msg.clone());

        loop {
            match self.socket.read_message() {
                Ok(Pong(_)) => {
                    thread::sleep(second);
                    self.ping();
                    self.send_message(msg.clone());
                },
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
        let second = Duration::from_secs(1);

        loop {
            match self.socket.read_message() {
                Ok(Pong(_)) => {thread::sleep(second); self.ping()},
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
        where F: Fn(Value) -> Option<T>
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
                        return Some(Err(GameWon));
                    } else if c == "game_lost" {
                        return Some(Err(GameLost));
                    } else if c == "game_update" {
                        // println!("{}", s);
                        return Some(Ok(diff_from_value(&arr[1])));
                    }
                }
            }
            None
        };

        self.get_message(f, Err(GameConnectionLost))
    }

    pub fn get_game_start(&mut self) -> Option<GameStart> {
        let f = |val| {
            if let Value::Array(arr) = val {
                if let Value::String(s) = &arr[0] {
                    if s == "game_start" {
                        return
                            serde_json::from_str(
                                &serde_json::to_string(&arr[1]).unwrap()
                            ).unwrap();
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
            return GameConnectionLost;
        }

        let mut tmp = self.get_diff();

<<<<<<< HEAD
        log!("Join match"; "type: {}", (start.game_type));
        log!("Join match"; "players: {}", (start.usernames.join(" | ")));
        log!("Join match"; "replay url: http://bot.generals.io/replays/{}", (start.replay_id));
=======
        println!("{}", Local::now().format("[ %T %D ]"));
        println!("Entering new {} match.", start.game_type);
        println!("Players: {}.", start.usernames.join(" | "));
        println!("Replay will be available at http://bot.generals.io/replays/{}", start.replay_id);
        player.init(start.playerIndex);
>>>>>>> parent of bedf2da (add action system)

        while let Ok(diff) = tmp {
            let mov = player.get_move(diff);

            self.send_move(mov);
            tmp = self.get_diff();
        }

        log!("Leave match"; "result: {:?}", (tmp.clone().unwrap_err()));
        return tmp.unwrap_err();
    }

    pub fn run_1v1(&mut self, mut player: &mut Box<dyn Player>) {
        loop {
            log!("Join 1v1"; "");
            self.join_1v1();

            if self.run_player(&mut player) == GameConnectionLost {
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

use std::result::Result;
use std::collections::HashMap;

fn diff_from_value(val: &Value) -> StateDiff {
    let update: GameUpdate =
        serde_json::from_str(&serde_json::to_string(val).unwrap()).unwrap();

    let helper = |o: Option<&Value>| {o.unwrap().as_i64().unwrap() as usize};

    let scores = update.scores
        .iter()
        .map(|m| (helper(m.get("i")), helper(m.get("tiles")), helper(m.get("total"))))
        .collect::<Vec<_>>();

    let mut scores2 = vec![(0, 0); scores.len()];

    for (i, land, armies) in scores {
        scores2[i] = (land, armies);
    }

    return StateDiff {
        turn: update.turn,
        map_diff: update.map_diff,
        cities_diff: update.cities_diff,
        generals: update.generals,
        scores: scores2
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
    pub usernames: Vec<String>
}

impl Drop for Client {
    fn drop(&mut self) {
        self.socket.close(None);

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
