use super::state::*;
use super::simulator::Player;

use std::{thread, time};

use tungstenite::*;
use tungstenite::Message::*;
use tungstenite::client::AutoStream;
use tungstenite::Error::*;

use serde::Deserialize;
use serde_json::{json, Value};

use chrono::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum GameErr {
    GameWon,
    GameLost,
    GameConnectionLost
}

pub use GameErr::*;

pub struct Client {
    socket: WebSocket<AutoStream>,
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
        let second = time::Duration::from_millis(1000);

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
        let second = time::Duration::from_millis(1000);

        loop {
            match self.socket.read_message() {
                Ok(Pong(_)) => {thread::sleep(second); self.ping()},
                Err(_) | Ok(Close(_)) => break,
                Ok(Text(s)) => {
                    if let Ok(val) = serde_json::from_str::<Value>(&s[2..]) {
                        println!("{}", val);
                    } else {
                        println!("Ok({:?})", s);
                    }
                    println!();
                }
                msg => {
                    println!("{:?}", msg);
                    println!();
                }
            }
        }
    }

    pub fn send_move(&mut self, mov: Option<Move>) {
        let mov = mov.unwrap_or(Move::new(0, 0, false));

        self.send_message(json!(["attack", mov.start, mov.end, mov.is50]));
    }

    pub fn get_diff(&mut self) -> Result<StateDiff, GameErr> {
        loop {
            match self.socket.read_message() {
                Err(_) | Ok(Close(_)) => return Err(GameConnectionLost),
                Ok(Text(s)) => {
                    if let Ok(Value::Array(arr)) = serde_json::from_str(&s[2..]) {
                        if let Value::String(c) = &arr[0] {
                            if c == "game_won" {
                                return Err(GameWon);
                            } else if c == "game_lost" {
                                return Err(GameLost);
                            } else if c == "game_update" {
                                // println!("{}", s);
                                return Ok(diff_from_value(&arr[1]));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn get_game_start(&mut self) -> Option<GameStart> {
        loop {
            let msg = self.socket.read_message();
            // println!("{:?}", msg);
            match msg {
                Err(_) | Ok(Close(_)) => return None,
                Ok(Text(s)) => {
                    if let Ok(Value::Array(arr)) = serde_json::from_str(&s[2..]) {
                        if let Value::String(s) = &arr[0] {
                            if s == "game_start" {
                                return
                                    serde_json::from_str(
                                        &serde_json::to_string(&arr[1]).unwrap()
                                    ).unwrap();
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn run_player(&mut self, player: &mut Box<dyn Player>) -> GameErr {
        let start;

        if let Some(s) = self.get_game_start() {
            start = s;
        } else {
            return GameConnectionLost;
        }

        let mut tmp = self.get_diff();

        println!("{}", Local::now().format("[ %T %D ]"));
        println!("Entering new {} match.", start.game_type);
        println!("Players: {}.", start.usernames.join(" | "));
        println!("Replay will be available at http://bot.generals.io/replays/{}", start.replay_id);
        player.init(start.playerIndex);

        while let Ok(diff) = tmp {
            let mov = player.get_move(diff);

            self.send_move(mov);
            tmp = self.get_diff();
        }

        println!("Result: {:?}", tmp.clone().unwrap_err());
        println!();
        return tmp.unwrap_err();
    }

    pub fn run_1v1(&mut self, mut player: &mut Box<dyn Player>) {
        loop {
            self.join_1v1();

            if self.run_player(&mut player) == GameConnectionLost {
                self.socket = connect(&self.url).expect("Unable to connect").0;
            }
        }
    }
}

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
