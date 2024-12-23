use crate::othello_board::*;
use rand::prelude::*;
use std::collections::VecDeque;
use tch::{self, Kind};
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

pub struct DQNAgent {
    vs_q: VarStore,
    vs_t: VarStore,
    q_network: Sequential,
    target_network: Sequential,
    optimizer: nn::Optimizer,
    memory: VecDeque<([BoardType; 64], i64, f64, [BoardType; 64], bool)>,
    gamma: f32,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    batch_size: usize,
    steps: usize,
    update_target_every: usize,
    rng: ThreadRng,
}
impl DQNAgent {
    pub fn new(
        state_dim: i64,
        action_dim: i64,
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
        let mut vs_q: VarStore = VarStore::new(Device::cuda_if_available());
        let mut vs_t: VarStore = VarStore::new(Device::cuda_if_available());

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
        vs_q.bfloat16();
        vs_t.bfloat16();
        let optimizer = nn::Adam::default().build(&vs_q, learning_rate).unwrap();
        let memory: VecDeque<([BoardType; 64], i64, f64, [BoardType; 64], bool)> =
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
            steps: 0,
            update_target_every,
            rng: rand::thread_rng(),
        }
    }
    pub fn remember(
        &mut self,
        state: [BoardType; 64],
        action: i64,
        reward: f64,
        next_state: [BoardType; 64],
        done: bool,
    ) {
        if self.memory.len() >= self.memory.capacity() {
            self.memory.remove(0);
        }
        self.memory
            .push_back((state, action, reward, next_state, done));
    }
    pub fn act(&mut self, state: [BoardType; 64], board: &Board) -> usize {
        let legal_actions: Vec<usize> = board
            .get_valid_moves(if board.black_turn { BLACK } else { WHITE })
            .iter()
            .map(|&i| i[0] * 8 + i[1])
            .collect();

        if self.rng.gen::<f64>() <= self.epsilon {
            *legal_actions.iter().choose(&mut self.rng).unwrap()
        } else {
            let state_tensor: Tensor =
                Tensor::from_slice(&state.iter().map(|&i| i as f32).collect::<Vec<f32>>()).to_kind(Kind::BFloat16).to(Device::cuda_if_available());
            let q_values = self.q_network.forward(&state_tensor);
            let legal_q_values: Tensor = q_values.index_select(
                0,
                &Tensor::from_slice(
                    &legal_actions
                        .iter()
                        .map(|&i| i as i64)
                        .collect::<Vec<i64>>(),
                ).to(Device::cuda_if_available()),
            );
            let idx: i64 = legal_q_values.argmax(0, false).try_into().unwrap();
            legal_actions[idx as usize]
        }
    }
    pub fn replay(&mut self) {
        if self.memory.len() >= self.batch_size as usize {
            let minibatch: Vec<&([BoardType; 64], i64, f64, [BoardType; 64], bool)> = self
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
                .forward(&states_tensor.to_kind(Kind::BFloat16).to(Device::cuda_if_available()))
                .gather(1, &actions_tensor.unsqueeze(1).to(Device::cuda_if_available()), false)
                .squeeze_dim(1);

            let next_q_values = self
                .target_network
                .forward(&next_states_tensor.to_kind(Kind::BFloat16).to(Device::cuda_if_available()))
                .max_dim(1, false)
                .0;

            let target_q_values =
                rewards_tensor.to_kind(Kind::BFloat16).to(Device::cuda_if_available()) + self.gamma * next_q_values * (dones_tensor.logical_not().to(Device::cuda_if_available()));
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
