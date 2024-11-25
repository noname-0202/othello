mod othello_board;
mod train;
use train::learn;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;


fn main() {
    const NUM_TRIALS: usize = 1000;
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
        .into_par_iter()
        .map(|_i| {
            let mut rng = rand::thread_rng();
            let learning_rate = rng.gen_range(0.0001..=0.01);
            let memory_maxlen = rng.gen_range(1..=10000);
            let gamma = rng.gen_range(0.01..=1.0);
            let epsilon = rng.gen_range(0.5..=1.0);
            let epsilon_min = rng.gen_range(0.01..=0.5);
            let epsilon_decay = rng.gen_range(0.99..=0.999);
            let batch_size = rng.gen_range(1..=10000);
            let update_target_every = rng.gen_range(1..=1000);
            let replay_every = rng.gen_range(1..=1000);
            let num_layers: u32 = rng.gen_range(1..=5);
            let first_layer_size: i64 = [8, 16, 32, 64, 128][rng.gen_range(0..5)];

            let win_rate = learn(
                100000,
                learning_rate,
                memory_maxlen,
                gamma,
                epsilon,
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
                    epsilon,
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
            || (0.0, (0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)), // 初期値
            |a, b| if a.0 > b.0 { a } else { b },                  // 最大値を選択
        );

    pb.finish_with_message("完了！");

    let (best_win_rate, best_params) = best_result;

    println!("最高の勝率: {}", best_win_rate);
    println!("最高のパラメータ:");
    println!("LEARNING_RATE: {}", best_params.0);
    println!("MEMORY_MAXLEN: {}", best_params.1);
    println!("GAMMA: {}", best_params.2);
    println!("EPSILON: {}", best_params.3);
    println!("EPSILON_MIN: {}", best_params.4);
    println!("EPSILON_DECAY: {}", best_params.5);
    println!("BATCH_SIZE: {}", best_params.6);
    println!("UPDATE_TARGET_EVERY: {}", best_params.7);
    println!("REPLAY_EVERY: {}", best_params.8);
    println!("NUM_LAYERS: {}", best_params.9);
    println!("FIRST_LAYER_SIZE: {}", best_params.10);
}