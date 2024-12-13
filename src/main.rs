mod agent;
mod othello_board;
use agent::DQNAgent;
use indicatif::{ProgressBar, ProgressStyle};
use othello_board::*;
use rand::prelude::*;
use std::time::Instant;

fn learn(
    num_epochs: i64,
    learning_rate: f64,
    memory_maxlen: usize,
    gamma: f32,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    batch_size: usize,
    update_target_every: usize,
    replay_every: usize,
    num_layers: u32,
    first_layer_size: i64,
) -> f64 {
    let mut board: Board = Board::new();
    let mut agent = DQNAgent::new(
        64,
        64,
        learning_rate,
        memory_maxlen,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size,
        update_target_every,
        num_layers,
        first_layer_size,
    );

    let mut results: [u64; 3] = [0, 0, 0];

    let mut ai_color: BoardType;
    let mut other_color: BoardType;
    let mut step: usize = 0;
    let mut place: [usize; 2];
    let mut other_place: [usize; 2];
    let mut observation: [BoardType; 64];
    let mut action: usize;
    let mut reward: f64 = 0.0;
    let mut observation_next: [BoardType; 64];
    let mut done: bool;
    let [mut whites, mut blacks, _]: [u8; 3];
    let bench_from: i64 = num_epochs - 101;
    let mut rng = rand::thread_rng();

    for epoch in 0..num_epochs {
        [ai_color, other_color] = if rng.gen::<bool>() {
            [BLACK, WHITE]
        } else {
            [WHITE, BLACK]
        };
        if other_color == BLACK {
            other_place = *board.get_valid_moves(other_color).choose(&mut rng).unwrap();
            board.apply_move(other_place);
        };
        while !board.game_ended() {
            observation = board.board.as_flattened().try_into().unwrap();

            action = agent.act(observation, &board);
            place = [action / 8, action % 8];
            board.apply_move(place);

            loop {
                done = board.game_ended();
                if (!done)
                    && (if other_color == BLACK {
                        board.black_turn
                    } else {
                        !board.black_turn
                    })
                {
                    other_place = *board.get_valid_moves(other_color).choose(&mut rng).unwrap();
                    board.apply_move(other_place);
                } else {
                    break;
                }
            }

            if done {
                [whites, blacks, _] = board.count_stones();
                if (ai_color == BLACK && blacks > whites) || (ai_color == WHITE && blacks < whites)
                {
                    reward = 1.0;
                } else if (ai_color == BLACK && blacks < whites)
                    || (ai_color == WHITE && blacks > whites)
                {
                    reward = -1.0;
                } else {
                    reward = 0.0;
                }
            } else {
                reward = 0.0;
            }
            observation_next = board.board.as_flattened().try_into().unwrap();
            agent.remember(observation, action as i64, reward, observation_next, done);
            step += 1;
            if step % replay_every == 0 {
                agent.replay();
            }
        }
        if epoch > bench_from {
            if reward == 1.0 {
                results[0] += 1;
            } else if reward == -1.0 {
                results[1] += 1;
            } else {
                results[2] += 1;
            }
        }
        board.reset();
    }
    (results[0] as f64) / (results.iter().sum::<u64>() as f64)
}
fn main() {
    let start: Instant = Instant::now();
    println!(
        "勝率: {:.3}",
        learn(1000, 0.0001, 100, 0.99, 1.0, 0.1, 0.995, 64, 100, 20, 5, 32)
    );
    /*
    const NUM_TRIALS: usize = 1000;

    let pb = ProgressBar::new(NUM_TRIALS as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let best_result = (0..NUM_TRIALS)
        .into_iter()
        .map(|_i| {
            let mut rng = rand::thread_rng();
            let learning_rate = rng.gen_range(0.0001..=0.001);
            let memory_maxlen = rng.gen_range(10..=500);
            let gamma = rng.gen_range(0.01..=1.0);
            let epsilon_min = rng.gen_range(0.1..=0.5);
            let epsilon_decay = rng.gen_range(0.99..=0.999);
            let batch_size = rng.gen_range(10..=500);
            let update_target_every = rng.gen_range(100..=200);
            let replay_every = rng.gen_range(5..=100);
            let num_layers: u32 = rng.gen_range(1..=4);
            let first_layer_size: i64 = [8, 16, 32, 64][rng.gen_range(0..5)];

            let win_rate = learn(
                10000,
                learning_rate,
                memory_maxlen,
                gamma,
                1.0,
                epsilon_min,
                epsilon_decay,
                batch_size,
                update_target_every,
                replay_every,
                num_layers,
                first_layer_size,
            );

            pb.inc(1);

            // 勝率とパラメータをタプルで返す
            (
                win_rate,
                (
                    learning_rate,
                    memory_maxlen,
                    gamma,
                    epsilon_min,
                    epsilon_decay,
                    batch_size,
                    update_target_every,
                    replay_every,
                    num_layers,
                    first_layer_size,
                ),
            )
        })
        .reduce(
            |a, b| if a.0 > b.0 { a } else { b },
        );

    pb.finish_with_message("完了！");

    let (best_win_rate, best_params) = best_result.unwrap();

    println!("最高の勝率: {}", best_win_rate);
    println!("最高のパラメータ:");
    println!("LEARNING_RATE: {}", best_params.0);
    println!("MEMORY_MAXLEN: {}", best_params.1);
    println!("GAMMA: {}", best_params.2);
    println!("EPSILON_MIN: {}", best_params.3);
    println!("EPSILON_DECAY: {}", best_params.4);
    println!("BATCH_SIZE: {}", best_params.5);
    println!("UPDATE_TARGET_EVERY: {}", best_params.6);
    println!("REPLAY_EVERY: {}", best_params.7);
    println!("NUM_LAYERS: {}", best_params.8);
    println!("FIRST_LAYER_SIZE: {}", best_params.9);*/

    let end = start.elapsed().as_secs();
    println!("実行時間:{}秒", end);
}
