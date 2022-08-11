use super::*;

use std::fs::{self, File};

use anyhow::Result;
use rand::{thread_rng, seq::SliceRandom};

use tch::Tensor;

fn train_from_replay(replay: &Replay, nn: &mut NN) {
    let players = replay.num_players();

    let (mut sim, turns) = replay.to_simulator();
    let mut feature_gens = vec![FeatureGen::new(); players];

    let mut features = Vec::new();
    let mut expected = Vec::new();

    let width  = sim.state.width ;
    let height = sim.state.height;

    for _ in 0..turns {
        let moves = sim.get_moves();

        for player in 0..players {
            if let Some(mov) = moves[player] {
                if sim.state.move_is_valid(player, mov) {
                    feature_gens[player].update(&sim.player_states[player]);

                    features.push(feature_gens[player].generate_features(player, nn.device()));
                    expected.push(tensor_of_move(mov, width, height).to_device(nn.device()));
                }
            }
        }

        sim.step();
        if sim.game_over().is_some() {
            break;
        }
    }

    if features.len() > 0 {
        nn.train(&Tensor::stack(&features, 0), &Tensor::stack(&expected, 0), 1);
    }
}

pub fn train_from_replays(replay_dir: &str, net_file_prefix: &str, epochs: usize, nn: &mut NN) -> Result<()> {
    let mut test_files = fs::read_dir(replay_dir)?
        .filter(|e| e.is_ok() && e.as_ref().unwrap().file_type().unwrap().is_file())
        .map(|e| e.unwrap().path().into_os_string().into_string().unwrap())
        .collect::<Vec<_>>();

    // let mut files2 = Vec::new();

    // println!("Filtering files with more than two players...");
    // for i in 50000..test_files.len() {
        // print!("{:6} {}", i, test_files[i]);
        // if Replay::read_from_file(&test_files[i]).ok().as_ref().map(Replay::num_players) == Some(2) {
            // files2.push(test_files[i].clone());
            // println!("");
        // } else {
            // println!(" asdf");
        // }
    // }

    // test_files = files2;

    let mut files = test_files.split_off(100);

    let mut rng = thread_rng();

    for epoch in 0..epochs {
        files.shuffle(&mut rng);

        let n_examples = files.len();

        for (i, filename) in files.iter_mut().enumerate() {
            println!("epoch {} example {}/{}", epoch, i, n_examples);

            println!("{}", filename);
            let replay = Replay::read_from_file(filename)?;

            train_from_replay(&replay, nn);

            if i % 100 == 0 && i > 0 {
                nn.to_writer(&mut File::create(&format!("{}_{}_{}.gio_nn", net_file_prefix, epoch, i))?)?;
            }
        }
    }

    Ok(())
}
