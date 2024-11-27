pub const EMPTY: i64 = 0;
pub const BLACK: i64 = 1;
pub const WHITE: i64 = 2;
const LOOKUP_ARRAY: [[isize; 2]; 8] = [
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
];

pub struct Board {
    default_board: [[i64; 8]; 8],
    pub board: [[i64; 8]; 8],
    pub black_turn: bool,
}

impl Board {
    pub fn new() -> Self {
        let mut default_board: [[i64; 8]; 8] = [[EMPTY; 8]; 8];
        default_board[3][4] = BLACK;
        default_board[4][3] = BLACK;
        default_board[3][3] = WHITE;
        default_board[4][4] = WHITE;

        Board {
            default_board,
            board: default_board.clone(),
            black_turn: true,
        }
    }

    pub fn reset(&mut self) {
        self.board = self.default_board.clone();
    }

    pub fn get_valid_moves(&self, color: i64) -> Vec<[usize; 2]> {
        let mut places = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                if self.board[i][j] == color {
                    if let Some(looked_up) = self.lookup(i as isize, j as isize, color) {
                        places.extend(looked_up);
                    }
                }
            }
        }
        places.sort_unstable();
        places.dedup();

        places
    }

    pub fn lookup(&self, row: isize, column: isize, color: i64) -> Option<Vec<[usize; 2]>> {
        let other: i64 = if color == BLACK { WHITE } else { BLACK };

        let mut places: Vec<[usize; 2]> = vec![];

        if row < 0 || row > 7 || column < 0 || column > 7 {
            return None;
        }

        for [x, y] in LOOKUP_ARRAY {
            let pos = self.check_direction(row, column, x, y, other);
            if pos.is_some() {
                places.push(pos.unwrap());
            }
        }
        Some(places)
    }
    pub fn check_direction(
        &self,
        row: isize,
        colmn: isize,
        row_add: isize,
        column_add: isize,
        other_color: i64,
    ) -> Option<[usize; 2]> {
        let mut i: isize = row + row_add;
        let mut j: isize = colmn + column_add;
        if i >= 0 && j >= 0 && i < 8 && j < 8 && self.board[i as usize][j as usize] == other_color {
            i += row_add;
            j += column_add;
            while i >= 0
                && j >= 0
                && i < 8
                && j < 8
                && self.board[i as usize][j as usize] == other_color
            {
                i += row_add;
                j += column_add;
            }
            if i >= 0 && j >= 0 && i < 8 && j < 8 && self.board[i as usize][j as usize] == EMPTY {
                Some([i as usize, j as usize])
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn apply_move(&mut self, move_: [usize; 2]) {
        let color: i64 = if self.black_turn { BLACK } else { WHITE };
        if self.get_valid_moves(color).contains(&move_) {
            self.board[move_[0]][move_[1]] = color;
            for i in 1usize..9usize {
                self.flip(i, move_, color);
            }
            if self
                .get_valid_moves(if self.black_turn { WHITE } else { BLACK })
                .len()
                > 0
            {
                self.black_turn = !self.black_turn;
            }
        }
    }
    pub fn flip(&mut self, direction: usize, position: [usize; 2], color: i64) {
        let [row_inc, col_inc] = LOOKUP_ARRAY[direction - 1];

        let mut i: isize = position[0] as isize + row_inc;
        let mut j: isize = position[1] as isize + col_inc;

        let other: i64 = if color == WHITE { BLACK } else { WHITE };

        if 0 <= i && i < 8 && 0 <= j && j < 8 && self.board[i as usize][j as usize] == other {
            let mut places: Vec<[isize; 2]> = vec![[i, j]];
            i += row_inc;
            j += col_inc;
            while 0 <= i && i < 8 && 0 <= j && j < 8 && self.board[i as usize][j as usize] == other
            {
                places.push([i, j]);
                i += row_inc;
                j += col_inc;
            }
            if 0 <= i && i < 8 && 0 <= j && j < 8 && self.board[i as usize][j as usize] == color {
                for [pos1, pos2] in places.iter() {
                    self.board[*pos1 as usize][*pos2 as usize] = color;
                }
            }
        }
    }
    pub fn game_ended(&self) -> bool {
        let [whites, blacks, empty] = self.count_stones();
        (whites == 0 || blacks == 0 || empty == 0)
            || (self.get_valid_moves(WHITE).len() == 0 && self.get_valid_moves(BLACK).len() == 0)
    }
    pub fn count_stones(&self) -> [u8; 3] {
        /*white, black, empty */
        let mut stones: [u8; 3] = [0, 0, 0];
        for i in self.board {
            for j in i {
                match j {
                    WHITE => stones[0] += 1,
                    BLACK => stones[1] += 1,
                    EMPTY => stones[2] += 1,
                    _ => (),
                }
            }
        }
        stones
    }
}
