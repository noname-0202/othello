mod agent;
mod othello_board;
use agent::DQNAgent;
use othello_board::*;
use rand::Rng;
use rand::seq::SliceRandom;
use std::io::{stdout, Write};
use indicatif::{ProgressBar, ProgressStyle};
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
    print_every: i64,
    replay_every: usize,
    num_layers: u32,
    first_layer_size: i64,
) -> f64 {
    let mut board = Board::new();
    let mut agent = DQNAgent::new(
        64,
        64,
        &mut board,
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

    let mut ai_color: i64;
    let mut other_color: i64;
    let mut step: usize;
    let mut place: [usize; 2];
    let mut other_place: [usize; 2];
    let mut observation: [i64; 64];
    let mut action: usize;
    let mut reward: f64;
    let mut observation_next: [i64; 64];
    let mut done: bool;
    let [mut whites, mut blacks, _]: [u8; 3];
    let mut rng = rand::thread_rng();
    const BLACK_AND_WHITE: [i64; 2] = [BLACK, WHITE];

    //#[inline]
    fn converter(board: &mut Board, color: i64) -> [i64; 64] {
        let mut converted: [i64; 64] = board.board.as_flattened().to_vec().try_into().unwrap();
        for [y, x] in board.get_valid_moves(color).iter() {
            converted[y * 8 + x] = 3;
        }
        converted
    }

    for epoch in 0..num_epochs {
        ai_color = *BLACK_AND_WHITE.choose(&mut rng).unwrap();
        other_color = if ai_color == WHITE { BLACK } else { WHITE };
        step = 0;
        if other_color == BLACK {
            other_place = *agent
                .board
                .get_valid_moves(ai_color)
                .choose(&mut rng)
                .unwrap();
            agent.board.apply_move(other_place);
        };
        while !agent.board.game_ended() {
            observation = converter(agent.board, ai_color);

            action = agent.act(observation);
            place = [
                (action as isize / 8) as usize,
                (action as isize % 8) as usize,
            ];
            agent.board.apply_move(place);

            while (!agent.board.game_ended())
                && (if other_color == BLACK {
                    agent.board.black_turn
                } else {
                    !agent.board.black_turn
                })
            {
                other_place = *agent
                    .board
                    .get_valid_moves(other_color)
                    .choose(&mut rng)
                    .unwrap();
                agent.board.apply_move(other_place);
            }
            if agent.board.game_ended() {
                [whites, blacks, _] = agent.board.count_stones();
                if (ai_color == BLACK && blacks > whites) || (ai_color == WHITE && blacks < whites)
                {
                    reward = 1.0;
                    results[0] += 1;
                } else if (ai_color == BLACK && blacks < whites)
                    || (ai_color == WHITE && blacks > whites)
                {
                    reward = -1.0;
                    results[1] += 1;
                } else {
                    reward = 0.0;
                    results[2] += 1;
                }
            } else {
                reward = 0.0;
            }
            observation_next = converter(agent.board, ai_color);
            done = agent.board.game_ended();
            agent.remember(observation, action as i64, reward, observation_next, done);
            step += 1;
            if step % replay_every == 0 {
                agent.replay();
            }
        }
        agent.board.reset();
        if (epoch + 1) % print_every == 0 {
            print!(
                "\r勝利: {}, 敗北: {}, 引き分け: {}, 勝率: {:.3}",
                results[0],
                results[1],
                results[2],
                (results[0] as f64) / (results.iter().sum::<u64>() as f64)
            );
            stdout().flush().unwrap();
        }
    }
    (results[0] as f64) / (results.iter().sum::<u64>() as f64)
}

fn main() {
    let start=Instant::now();
    learn(5000, 0.0001, 100, 0.99, 1.0, 0.2, 0.995, 64, 100, 1000000, 20, 2, 32);
    /*
    const NUM_TRIALS: usize = 5;
    // 並列処理を行うスレッドプールを指定されたジョブ数で作成

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
            let memory_maxlen = rng.gen_range(100..=500);
            let gamma = rng.gen_range(0.01..=1.0);
            let epsilon_min = rng.gen_range(0.1..=0.5);
            let epsilon_decay = rng.gen_range(0.99..=0.999);
            let batch_size = rng.gen_range(50..=200);
            let update_target_every = rng.gen_range(100..=200);
            let replay_every = rng.gen_range(1..=100);
            let num_layers: u32 = rng.gen_range(1..=4);
            let first_layer_size: i64 = [8, 16, 32, 64, 128][rng.gen_range(0..5)];

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
                1000000,
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
    println!("FIRST_LAYER_SIZE: {}", best_params.9);
    */

    let end=start.elapsed().as_secs();
    println!("実行時間:{}秒",end);
}
