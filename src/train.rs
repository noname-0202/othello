use crate::othello_board::*;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use std::collections::VecDeque;
use std::io::{stdout, Write};
use tch;
use tch::nn::{Module, OptimizerConfig, Sequential, VarStore};
use tch::{nn, Device, Tensor};

fn dqn(
    vs: &nn::Path,
    input_dim: i64,
    output_dim: i64,
    num_layers: u32,
    first_layer_size: i64,
) -> Sequential {
    (0..num_layers)
        .fold(
            nn::seq()
                .add(nn::linear(
                    vs / "layer_in",
                    input_dim,
                    first_layer_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu()),
            |seq, i| {
                seq.add(nn::linear(
                    vs / format!("layer_{}", i),
                    first_layer_size * (2i64.pow(i)),
                    first_layer_size * (2i64.pow(i + 1)),
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
            },
        )
        .add(nn::linear(
            vs / "layer_out",
            first_layer_size * (2i64.pow(num_layers)),
            output_dim,
            Default::default(),
        ))
}

struct DQNAgent<'a> {
    vs_q: VarStore,
    vs_t: VarStore,
    q_network: Sequential,
    target_network: Sequential,
    optimizer: nn::Optimizer,
    memory: VecDeque<([i64; 64], i64, f64, [i64; 64], bool)>,
    gamma: f32,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    batch_size: usize,
    board: &'a mut Board,
    steps: usize,
    rng: rand::rngs::ThreadRng,
    update_target_every: usize,
}

impl<'a> DQNAgent<'a> {
    fn new(
        state_dim: i64,
        action_dim: i64,
        board: &'a mut Board,
        learning_rate: f64,
        memory_maxlen: usize,
        gamma: f32,
        epsilon: f64,
        epsilon_min: f64,
        epsilon_decay: f64,
        batch_size: usize,
        update_target_every: usize,
        num_layers: u32,
        first_layer_size: i64,
    ) -> Self {
        let vs_q: VarStore = VarStore::new(Device::Cpu);
        let mut vs_t: VarStore = VarStore::new(Device::Cpu);
        let q_network: Sequential = dqn(
            &vs_q.root(),
            state_dim,
            action_dim,
            num_layers,
            first_layer_size,
        );
        let target_network: Sequential = dqn(
            &vs_t.root(),
            state_dim,
            action_dim,
            num_layers,
            first_layer_size,
        );
        let _ = vs_t.copy(&vs_q);
        let optimizer = nn::Adam::default().build(&vs_q, learning_rate).unwrap();
        let memory: VecDeque<([i64; 64], i64, f64, [i64; 64], bool)> =
            VecDeque::with_capacity(memory_maxlen);
        Self {
            vs_q,
            vs_t,
            q_network,
            target_network,
            optimizer,
            memory,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            batch_size,
            board,
            steps: 0,
            rng: rand::thread_rng(),
            update_target_every,
        }
    }
    fn remember(
        &mut self,
        state: [i64; 64],
        action: i64,
        reward: f64,
        next_state: [i64; 64],
        done: bool,
    ) {
        if self.memory.len() >= self.memory.capacity() {
            self.memory.remove(0);
        }
        self.memory
            .push_back((state, action, reward, next_state, done));
    }
    fn get_legal_actions(&mut self) -> Vec<usize> {
        let mut legal_actions: Vec<usize> = Vec::new();
        for [pos1, pos2] in
            self.board
                .get_valid_moves(if self.board.black_turn { BLACK } else { WHITE })
        {
            legal_actions.push(pos1 * 8 + pos2);
        }
        legal_actions
    }
    fn act(&mut self, state: [i64; 64]) -> usize {
        let legal_actions: Vec<usize> = self.get_legal_actions();
        if self.rng.gen::<f64>() <= self.epsilon {
            *legal_actions.choose(&mut self.rng).unwrap()
        } else {
            let state_tensor: Tensor =
                Tensor::from_slice(&state.iter().map(|&i| i as f32).collect::<Vec<f32>>());
            let q_values = self.q_network.forward(&state_tensor);
            let legal_q_values: Tensor = q_values.index_select(
                0,
                &Tensor::from_slice(
                    legal_actions
                        .iter()
                        .map(|&i| i as i64)
                        .collect::<Vec<i64>>()
                        .as_slice(),
                ),
            );
            let idx: i64 = legal_q_values.argmax(0, false).try_into().unwrap();
            legal_actions[idx as usize]
        }
    }
    fn replay(&mut self) {
        if self.memory.len() >= self.batch_size as usize {
            let minibatch: Vec<&([i64; 64], i64, f64, [i64; 64], bool)> = self
                .memory
                .iter()
                .choose_multiple(&mut self.rng, self.batch_size);
            let (states, actions, rewards, next_states, dones): (
                Vec<[f32; 64]>,
                Vec<i64>,
                Vec<f32>,
                Vec<[f32; 64]>,
                Vec<bool>,
            ) = minibatch.iter().fold(
                (
                    Vec::with_capacity(self.batch_size),
                    Vec::with_capacity(self.batch_size),
                    Vec::with_capacity(self.batch_size),
                    Vec::with_capacity(self.batch_size),
                    Vec::with_capacity(self.batch_size),
                ),
                |(mut state, mut action, mut reward, mut next_state, mut done),
                 &(s, a, r, n, d)| {
                    state.push({
                        let tmp: [f32; 64] = s
                            .iter()
                            .map(|&i| i as f32)
                            .collect::<Vec<f32>>()
                            .try_into()
                            .unwrap();
                        tmp
                    });
                    action.push(*a);
                    reward.push(*r as f32);
                    next_state.push({
                        let tmp: [f32; 64] = n
                            .iter()
                            .map(|&i| i as f32)
                            .collect::<Vec<f32>>()
                            .try_into()
                            .unwrap();
                        tmp
                    });
                    done.push(*d);
                    (state, action, reward, next_state, done)
                },
            );

            let states_tensor: Tensor = Tensor::from_slice2(&states);
            let actions_tensor: Tensor = Tensor::from_slice(&actions);
            let rewards_tensor: Tensor = Tensor::from_slice(&rewards);
            let next_states_tensor: Tensor = Tensor::from_slice2(&next_states);
            let dones_tensor: Tensor = Tensor::from_slice(&dones);

            let q_values: Tensor = self
                .q_network
                .forward(&states_tensor)
                .gather(1, &actions_tensor.unsqueeze(1), false)
                .squeeze_dim(1);

            let next_q_values = self
                .target_network
                .forward(&next_states_tensor)
                .max_dim(1, false)
                .0;

            let target_q_values =
                rewards_tensor + self.gamma * next_q_values * (dones_tensor.logical_not());
            let loss = q_values.mse_loss(&target_q_values, tch::Reduction::Mean);

            self.optimizer.zero_grad();
            self.optimizer.backward_step_clip_norm(&loss, 1.0);

            self.steps += 1;
            if self.steps % self.update_target_every == 0 {
                let _ = self.vs_t.copy(&self.vs_q);
            }

            if self.epsilon > self.epsilon_min {
                self.epsilon *= self.epsilon_decay;
            }
        }
    }
}

pub fn learn(
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
