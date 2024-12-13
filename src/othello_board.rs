pub type BoardType=i32;
pub const EMPTY: BoardType = 0;
pub const BLACK: BoardType = 1;
pub const WHITE: BoardType = 2;
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
    pub board: [[BoardType; 8]; 8],
    pub black_turn: bool,
}

impl Board {
    pub fn new() -> Self {
        let mut board: [[BoardType; 8]; 8] = [[EMPTY; 8]; 8];
        board[3][4] = BLACK;
        board[4][3] = BLACK;
        board[3][3] = WHITE;
        board[4][4] = WHITE;

        Board {
            board,
            black_turn: true,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    pub fn get_valid_moves(&self, color: BoardType) -> Vec<[usize; 2]> {
        let mut valid_moves: Vec<[usize; 2]> = Vec::new();
    
        for i in 0..8 {
            for j in 0..8 {
                if self.board[i][j] == EMPTY {
                    if self.lookup(i as isize, j as isize, color).is_some() {
                        valid_moves.push([i, j]);
                    }
                }
            }
        }
        valid_moves
    }
    pub fn lookup(&self, row: isize, column: isize, color: BoardType) -> Option<Vec<[usize; 2]>> {
        let other: BoardType = if color == BLACK { WHITE } else { BLACK };

        let mut places: Vec<[usize; 2]> = vec![];

        if row < 0 || row > 7 || column < 0 || column > 7 {
            return None;
        }

        for [x, y] in LOOKUP_ARRAY {
            let mut i: isize = row + x;
            let mut j: isize = column + y;
            if i >= 0 && j >= 0 && i < 8 && j < 8 && self.board[i as usize][j as usize] == other {
                i += x;
                j += y;
                while i >= 0
                    && j >= 0
                    && i < 8
                    && j < 8
                    && self.board[i as usize][j as usize] == other
                {
                    i += x;
                    j += y;
                }
                if i >= 0 && j >= 0 && i < 8 && j < 8 && self.board[i as usize][j as usize] == EMPTY
                {
                    places.push([i as usize, j as usize]);
                }
            };
        }
        Some(places)
    }

    pub fn apply_move(&mut self, move_: [usize; 2]) {
        let color: BoardType = if self.black_turn { BLACK } else { WHITE };
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
    pub fn flip(&mut self, direction: usize, position: [usize; 2], color: BoardType) {
        let [row_inc, col_inc] = LOOKUP_ARRAY[direction - 1];

        let mut i: isize = position[0] as isize + row_inc;
        let mut j: isize = position[1] as isize + col_inc;

        let other: BoardType = if color == WHITE { BLACK } else { WHITE };

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
