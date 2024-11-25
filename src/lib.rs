mod othello_board;
mod agent;
use othello_board::*;
use agent::DQNAgent;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use std::io::{stdout, Write};

#[pyfunction]
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
) -> PyResult<f64> {
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
    Ok((results[0] as f64) / (results.iter().sum::<u64>() as f64))
}

#[pymodule]
fn othello(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(learn, m)?)?;
    Ok(())
}
