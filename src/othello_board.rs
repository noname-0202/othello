pub const EMPTY: i64 = 0;
pub const BLACK: i64 = 1;
pub const WHITE: i64 = 2;

pub struct Board {
    default_board: [[i64; 8]; 8],
    pub board: [[i64; 8]; 8],
    lookup_array: [[isize; 2]; 8],
    pub black_turn: bool,
}
impl Board {
    pub fn new() -> Self {
        let mut default_board: [[i64; 8]; 8] = [[EMPTY; 8]; 8];
        default_board[3][4] = BLACK;
        default_board[4][3] = BLACK;
        default_board[3][3] = WHITE;
        default_board[4][4] = WHITE;
        let lookup_array: [[isize; 2]; 8] = [
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
        ];
        Board {
            default_board,
            board: default_board.clone(),
            lookup_array,
            black_turn: true,
        }
    }
    pub fn reset(&mut self) {
        self.board = self.default_board.clone();
    }
    fn lookup(&mut self, row: isize, column: isize, color: i64) -> Option<Vec<[usize; 2]>> {
        let other: i64 = if color == BLACK { WHITE } else { BLACK };

        let mut places: Vec<[usize; 2]> = vec![];

        if row < 0 || row > 7 || column < 0 || column > 7 {
            return None;
        }

        for [x, y] in self.lookup_array.clone().iter() {
            let pos = self.check_direction(row, column, *x, *y, other);
            if pos.is_some() {
                places.push(pos.unwrap());
            }
        }
        Some(places)
    }
    fn check_direction(
        &mut self,
        row: isize,
        colmn: isize,
        row_add: isize,
        column_add: isize,
        other_color: i64,
    ) -> Option<[usize; 2]> {
        let mut i: isize = row + row_add;
        let mut j: isize = colmn + column_add;
        if i >= 0 && j >= 0 && i < 8 && j < 8 && self.board[i as usize][j as usize] == other_color
        {
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
    pub fn get_valid_moves(&mut self, color: i64) -> Vec<[usize; 2]> {
        let mut looked_up: Option<Vec<[usize; 2]>>;
        let mut places: Vec<[usize; 2]> = vec![];

        for i in 0..8 {
            for j in 0..8 {
                if self.board[i][j] == color {
                    looked_up = self.lookup(i as isize, j as isize, color);
                    if looked_up.is_some(){
                        places.append(looked_up.unwrap().as_mut());
                    }
                }
            }
        }
        places.sort();
        places.dedup();
        places
    }
    pub fn apply_move(&mut self, move_: [usize; 2]) {
        if self.game_ended() {
            panic!("The game has ended.");
        }

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
    fn flip(&mut self, direction: usize, position: [usize; 2], color: i64) {
        let [row_inc, col_inc] = self.lookup_array[direction - 1];

        let mut i: isize = position[0] as isize + row_inc;
        let mut j: isize = position[1] as isize + col_inc;

        let other: i64 = if color == WHITE { BLACK } else { WHITE };

        const ZERO_TO_SEVEN: [isize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

        if ZERO_TO_SEVEN.contains(&i)
            && ZERO_TO_SEVEN.contains(&j)
            && self.board[i as usize][j as usize] == other
        {
            let mut places: Vec<[isize; 2]> = vec![[i, j]];
            i += row_inc;
            j += col_inc;
            while ZERO_TO_SEVEN.contains(&i)
                && ZERO_TO_SEVEN.contains(&j)
                && self.board[i as usize][j as usize] == other
            {
                places.push([i, j]);
                i += row_inc;
                j += col_inc;
            }
            if ZERO_TO_SEVEN.contains(&i)
                && ZERO_TO_SEVEN.contains(&j)
                && self.board[i as usize][j as usize] == color
            {
                for [pos1, pos2] in places.iter() {
                    self.board[*pos1 as usize][*pos2 as usize] = color;
                }
            }
        }
    }
    pub fn game_ended(&mut self) -> bool {
        let [whites, blacks, empty] = self.count_stones();
        if (whites == 0 || blacks == 0 || empty == 0)
            || (self.get_valid_moves(BLACK).len() == 0 && self.get_valid_moves(WHITE).len() == 0)
        {
            true
        } else {
            false
        }
    }
    pub fn count_stones(&mut self) -> [u8; 3] {
        /*white, black, empty */
        let mut stones: [u8; 3] = [0, 0, 0];
        for i in 0..8 {
            for j in 0..8 {
                match self.board[i][j] {
                    WHITE => stones[0] += 1,
                    BLACK => stones[1] += 1,
                    EMPTY => stones[2] += 1,
                    _=>()
                }
            }
        }
        stones
    }
}
