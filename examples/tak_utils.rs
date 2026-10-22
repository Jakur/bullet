pub const STACK_DEPTH: usize = 10;
pub const HIDDEN_SIZE: usize = 512;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const WHITE_FLAT: ValidPiece = ValidPiece(0);
pub const BLACK_FLAT: ValidPiece = ValidPiece(1);
pub const WHITE_WALL: ValidPiece = ValidPiece(2);
pub const BLACK_WALL: ValidPiece = ValidPiece(3);
pub const WHITE_CAP: ValidPiece = ValidPiece(4);
pub const BLACK_CAP: ValidPiece = ValidPiece(5);
const _ASS: () = assert!(
    WHITE_FLAT.flip_color().0 == BLACK_FLAT.0
        && BLACK_WALL.flip_color().0 == WHITE_WALL.0
        && BLACK_CAP.flip_color().0 == WHITE_CAP.0
);

// pub static NNUE: Network = unsafe {
//     let bytes = include_bytes!("../checkpoints/test-240b/quantised.bin");
//     assert!(bytes.len() == std::mem::size_of::<Network>());
//     std::mem::transmute(*bytes)
// };

use anyhow::{anyhow, bail, Result};
use bullet_lib::default::loader::DataLoader;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
// use topaz_tak::board::Board6;
// use topaz_tak::eval::{build_nn_repr, BoardData, PieceSquare};
// use topaz_tak::TakBoard;
// use topaz_tak::{Color, GameMove, Piece, Position};
use zerocopy::native_endian::I16;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes};

const SUGGESTED_CHUNK_SIZE: usize = 0x2000;

const TAK_MAGIC: u32 = u32::from_be_bytes(*b"TAK6");

#[derive(Debug)]
pub struct CompressedTrainingDataEntryReader {
    chunk: Vec<u8>,
    input_file: CompressedTrainingDataFile,
    offset: usize,
    file_size: u64,
    is_end: bool,
    needs_new_chunk: bool,
}

impl<'a> CompressedTrainingDataEntryReader {
    pub fn new(path: &str) -> Result<Self> {
        let chunk = Vec::with_capacity(SUGGESTED_CHUNK_SIZE);

        let mut reader = Self {
            chunk,
            input_file: CompressedTrainingDataFile::new(path, false, false)?,
            offset: 0,
            file_size: std::fs::metadata(path)?.len(),
            is_end: false,
            needs_new_chunk: false,
        };

        if !reader.input_file.has_next_chunk() {
            reader.is_end = true;
            bail!("End of File");
        } else {
            reader.chunk = match reader.input_file.read_next_chunk() {
                Ok(chunk) => chunk,
                Err(e) => bail!(format!("Binpack Error: {e}")),
            };
        }

        Ok(reader)
    }

    /// Get the size of the file in bytes
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get how much of the file has been read so far
    pub fn read_bytes(&self) -> u64 {
        self.input_file.read_bytes()
    }

    /// Check if there are more TrainingDataEntry to read
    pub fn has_next(&'a self) -> bool {
        self.offset < self.chunk.len()
    }
    // /// Get the next TrainingDataEntry
    pub fn next(&'a self) -> (EntryReader<'a>, usize) {
        let end = self.offset + EntryHeader::SIZE;
        let header = EntryHeader::ref_from_bytes(&self.chunk[self.offset..end]).unwrap();
        let data_len = header.data_len as usize;
        let plies_len = header.plys_len as usize;
        let (psquares, rest) = <[PSquare]>::ref_from_prefix_with_elems(&self.chunk[end..], data_len).unwrap();
        let (plies, _) = <[MoveList]>::ref_from_prefix_with_elems(&rest, plies_len).unwrap();

        let full_size = EntryHeader::SIZE + header.plys_len as usize * MoveList::SIZE + header.data_len as usize;

        let entry = Entry::new(header, psquares, plies);
        (EntryReader::new(entry), full_size)
    }

    pub fn fetch_next_chunk_if_needed(&mut self) -> bool {
        if !self.has_next() {
            self.needs_new_chunk = false;
            if self.input_file.has_next_chunk() {
                self.chunk = self.input_file.read_next_chunk().unwrap();
                self.offset = 0;
            } else {
                self.is_end = true;
                return false;
            }
        }
        true
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ScoredPosition {
    pub(crate) data: BoardData,
    pub(crate) score: I16,
    pub(crate) result: u8,
}

impl ScoredPosition {
    fn new(data: BoardData, score: I16, result: u8) -> Self {
        Self { data, score, result }
    }
    fn symmetry(&mut self, rotation: usize) {
        self.data = self.data.symmetry(rotation);
    }
}

#[repr(C)]
#[derive(TryFromBytes, IntoBytes, KnownLayout, Immutable)]
struct ChunkHeader {
    magic: TakMagic,
    chunk_size: u32,
}

#[repr(C)]
#[derive(TryFromBytes, KnownLayout, Immutable)]
struct Chunk {
    header: ChunkHeader,
    body: [u8],
}

#[repr(u32)]
#[derive(TryFromBytes, IntoBytes, KnownLayout, Immutable)]
enum TakMagic {
    Tak = TAK_MAGIC,
}

#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug)]
struct EntryHeader {
    caps: [u8; 2],     // 0 - 36 for white, black capstone, else invalid
    white_to_move: u8, // 0 == False, else True
    extra_score: u8,   // Mostly for padding, use as needed
    score: I16,        // Side to move relative
    result: u8,
    komi: u8,
    data_len: u8, // In elements
    plys_len: u8, // In elements
}

#[derive(Debug, Clone, Copy)]
enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    fn wall(&self) -> ValidPiece {
        match self {
            Color::White => WHITE_WALL,
            Color::Black => BLACK_WALL,
        }
    }
    fn flat(&self) -> ValidPiece {
        match self {
            Color::White => WHITE_FLAT,
            Color::Black => BLACK_FLAT,
        }
    }
}

#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Clone, Copy)]
struct PSquare(u8);

impl From<PSquare> for PieceSquare {
    fn from(value: PSquare) -> Self {
        PieceSquare(value.0)
    }
}

impl EntryHeader {
    const SIZE: usize = std::mem::size_of::<Self>();
    fn is_white(&self) -> bool {
        self.white_to_move != 0
    }
}

pub fn build_piece_square(sq: usize, piece: ValidPiece) -> PieceSquare {
    PieceSquare::new(sq, piece.0)
}

#[derive(Debug)]
#[repr(C)]
struct Entry<'a> {
    header: &'a EntryHeader,
    psquares: &'a [PSquare],
    plies: &'a [MoveList],
}

impl<'a> Entry<'a> {
    fn new(header: &'a EntryHeader, psquares: &'a [PSquare], plies: &'a [MoveList]) -> Self {
        Self { header, psquares, plies }
    }
}

#[derive(Debug)]
pub struct EntryReader<'a> {
    entry: Entry<'a>,
    position: usize,
    cache: Option<BoardData>,
}

impl<'a> EntryReader<'a> {
    fn new(entry: Entry<'a>) -> Self {
        Self { entry, position: 0, cache: None }
    }
}

impl<'a> Iterator for EntryReader<'a> {
    type Item = ScoredPosition;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == 0 {
            let header = &self.entry.header;
            let len = header.data_len as usize;
            let mut out_data = [PieceSquare(255); 62];
            for i in 0..len {
                out_data[i] = self.entry.psquares[i].into();
            }
            let board = BoardData::new(header.caps.clone(), out_data, len as u8, header.is_white());
            self.cache = Some(board);
            self.position += 1;
            Some(ScoredPosition::new(board, header.score, header.result))
        } else {
            let idx = self.position - 1;
            let data = self.entry.plies.get(idx)?;
            let header = &self.entry.header;
            let mut last = self.cache.unwrap();
            let square = data.data & 63;
            let mut white_to_move = header.is_white();
            let mut result = header.result;
            if idx % 2 == 0 {
                result = 2 - result;
            } else {
                white_to_move = !white_to_move;
            }
            let color = if white_to_move { Color::White } else { Color::Black };
            let is_cap = (data.data & 0b1000_0000) == 0b1000_0000;
            let is_wall = (data.data & 0b0100_0000) == 0b0100_0000;
            let piece = if is_cap {
                if last.caps[0] >= 36 {
                    last.caps[0] = square;
                } else {
                    last.caps[1] = square;
                }
                if white_to_move {
                    WHITE_CAP
                } else {
                    BLACK_CAP
                }
            } else if is_wall {
                color.wall()
            } else {
                color.flat()
            };
            let piece_square: PieceSquare = build_piece_square(square.into(), piece);

            last.data[last.data_len as usize] = piece_square;
            last.data_len += 1;
            last.white_to_move = !last.white_to_move;
            self.cache = Some(last);
            self.position += 1;
            Some(ScoredPosition::new(last, data.score.into(), result))
        }
    }
}

#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug)]
struct MoveList {
    data: u8,
    extra_score: u8,
    score: I16,
}

impl MoveList {
    const SIZE: usize = std::mem::size_of::<Self>();
    // fn from_game_move(mv: GameMove, score: i16, extra_score: u8) -> Self {
    //     assert!(mv.is_place_move());
    //     let square = mv.src_index() as u8;
    //     let top = match mv.place_piece() {
    //         Piece::WhiteFlat | Piece::BlackFlat => 0,
    //         Piece::WhiteWall | Piece::BlackWall => 1,
    //         Piece::WhiteCap | Piece::BlackCap => 2,
    //     };
    //     let data = square | (top << 6);
    //     Self { data, score: score.into(), extra_score }
    // }
}

#[derive(Debug)]
pub struct CompressedTrainingDataFile {
    file: File,
    read_bytes: u64,
}

impl CompressedTrainingDataFile {
    pub fn new(path: &str, append: bool, create: bool) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(create).append(append).open(path)?;

        Ok(Self { file, read_bytes: 0 })
    }

    pub fn read_bytes(&self) -> u64 {
        self.read_bytes
    }

    pub fn has_next_chunk(&mut self) -> bool {
        if let Ok(pos) = self.file.stream_position() {
            if let Ok(len) = self.file.seek(SeekFrom::End(0)) {
                if self.file.seek(SeekFrom::Start(pos)).is_ok() {
                    return pos < len;
                }
            }
        }
        false
    }

    pub fn read_next_chunk(&mut self) -> Result<Vec<u8>> {
        const HEADER_SIZE: usize = std::mem::size_of::<ChunkHeader>();
        let mut buf = [0u8; HEADER_SIZE];
        self.file.read_exact(&mut buf)?;
        let result = ChunkHeader::try_read_from_bytes(&mut buf);
        let header = result.map_err(|e| e.map_src(|s| &[0; HEADER_SIZE]))?; // Todo figure out this mess

        self.read_bytes += HEADER_SIZE as u64;

        let chunk_size = header.chunk_size as u64;
        let mut data = vec![0u8; (chunk_size) as usize];

        // EBNF: Chain
        self.file.read_exact(&mut data)?;

        self.read_bytes += chunk_size as u64;
        Ok(data)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidPiece(pub u8);

impl ValidPiece {
    pub const fn without_color(self) -> u8 {
        self.0 >> 1
    }
    const fn flip_color(self) -> Self {
        Self(self.0 ^ 1) // Toggle bit 0
    }
    pub const fn promote_cap(self) -> Self {
        Self(self.0 | 4) // Set bit 2
    }
    pub const fn is_white(self) -> bool {
        (self.0 & 1) == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PieceSquare(pub u8);

impl PieceSquare {
    pub fn new(square: usize, piece: u8) -> Self {
        Self((square as u8) | piece << 6)
    }
    pub fn square(self) -> u8 {
        self.0 & 63
    }
    pub fn piece(self) -> ValidPiece {
        let masked = 0b1100_0000 & self.0;
        ValidPiece(masked >> 6)
    }
    pub fn promote_wall(&mut self) {
        self.0 |= 128;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BoardData {
    pub caps: [u8; 2],
    pub data: [PieceSquare; 62], // Each stack must be presented from top to bottom sequentially
    pub data_len: u8,
    pub white_to_move: bool,
}

impl BoardData {
    const SIZE: u8 = 6;
    const SYM_TABLE: [[u8; 36]; 8] = Self::build_symmetry_table();
    pub fn new(caps: [u8; 2], data: [PieceSquare; 62], data_len: u8, white_to_move: bool) -> Self {
        Self { caps, data, data_len, white_to_move }
    }
    pub fn symmetry(mut self, idx: usize) -> Self {
        if idx == 0 {
            return self;
        }
        assert!(idx < 8);
        let table = &Self::SYM_TABLE[idx];
        if self.caps[0] < 36 {
            self.caps[0] = table[self.caps[0] as usize];
        }
        if self.caps[1] < 36 {
            self.caps[1] = table[self.caps[1] as usize];
        }
        for i in 0..(self.data_len as usize) {
            let old = self.data[i];
            self.data[i] = PieceSquare::new(table[old.square() as usize] as usize, old.piece().0);
        }
        self
    }
    const fn build_symmetry_table() -> [[u8; 36]; 8] {
        [
            Self::transform(0),
            Self::transform(1),
            Self::transform(2),
            Self::transform(3),
            Self::transform(4),
            Self::transform(5),
            Self::transform(6),
            Self::transform(7),
        ]
    }
    const fn transform(rotation: usize) -> [u8; 36] {
        let mut data = [(0, 0); 36];
        let mut i = 0;
        while i < 36 {
            let (row, col) = Self::row_col(i as u8);
            data[i] = (row, col);
            i += 1;
        }
        match rotation {
            1 => Self::flip_ns(&mut data),
            2 => Self::flip_ew(&mut data),
            3 => Self::rotate(&mut data),
            4 => {
                Self::rotate(&mut data);
                Self::rotate(&mut data);
            }
            5 => {
                Self::rotate(&mut data);
                Self::rotate(&mut data);
                Self::rotate(&mut data);
            }
            6 => {
                Self::rotate(&mut data);
                Self::flip_ns(&mut data);
            }
            7 => {
                Self::rotate(&mut data);
                Self::flip_ew(&mut data);
            }
            _ => {}
        };
        let mut out = [0; 36];
        let mut i = 0;
        while i < 36 {
            let (row, col) = data[i];
            out[i] = Self::index(row, col);
            i += 1;
        }
        out
    }
    const fn flip_ns(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (row, _col) = &mut arr[i];
            *row = Self::SIZE - 1 - *row;
            i += 1;
        }
    }
    const fn flip_ew(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (_row, col) = &mut arr[i];
            *col = Self::SIZE - 1 - *col;
            i += 1;
        }
    }
    const fn rotate(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (row, col) = &mut arr[i];
            let new_row = Self::SIZE - 1 - *col;
            *col = *row;
            *row = new_row;
            i += 1;
        }
    }
    const fn row_col(index: u8) -> (u8, u8) {
        (index / Self::SIZE, index % Self::SIZE)
    }
    const fn index(row: u8, col: u8) -> u8 {
        row * Self::SIZE + col
    }
}

#[derive(Clone, Copy)]
pub struct TakSimple6 {}

impl TakSimple6 {
    pub const SQUARE_INPUTS: usize = 36 * (6 + 2 * 2 * STACK_DEPTH);
    const HALF: usize = Self::SQUARE_INPUTS / 2;
    // Squares + Side + Reserves
    pub const NUM_INPUTS: usize = TakSimple6::SQUARE_INPUTS + 8 + 80;

    pub fn handle_features<F: FnMut(usize, usize)>(&self, pos: &ScoredPosition, mut f: F) {
        let mut reserves: [usize; 2] = [31, 31];
        let mut top = 0usize;
        for (piece, square, depth_idx) in pos.into_iter() {
            let c = (piece.is_white() ^ pos.data.white_to_move) as usize; // 0 if matches, else 1
            reserves[c] -= 1;
            let location = if depth_idx == 0 {
                // top = piece.0;
                top = c; // Friendly or enemy top
                usize::from(piece.without_color()) // + depth, but we know it is 0
            } else {
                usize::from(depth_idx) + top
            };
            let sq = usize::from(square);

            let stm = [0, Self::HALF][c] + 36 * location + sq;
            let ntm = [Self::HALF, 0][c] + 36 * location + sq;
            f(stm, ntm);
        }
        // let white_res_adv = (31 + reserves[0] - reserves[1]).clamp(23, 39);
        // let black_res_adv = (31 + reserves[1] - reserves[0]).clamp(23, 39);
        if pos.data.white_to_move {
            // Am white, with reserves vs. Am black with reserves
            f(Self::SQUARE_INPUTS + reserves[0] / 2, Self::SQUARE_INPUTS + 40 + reserves[1] / 2);
            // f(Self::SQUARE_INPUTS, Self::SQUARE_INPUTS + 1);
        } else {
            // Am with reserves
            f(Self::SQUARE_INPUTS + 40 + reserves[0] / 2, Self::SQUARE_INPUTS + reserves[1] / 2);
        }
    }
}

impl IntoIterator for ScoredPosition {
    type Item = (ValidPiece, u8, u8);
    type IntoIter = TakBoardIter;
    fn into_iter(self) -> Self::IntoIter {
        TakBoardIter { board: self, idx: 0, last: u8::MAX, depth: 0 }
    }
}

pub struct TakBoardIter {
    board: ScoredPosition,
    idx: usize,
    last: u8,
    depth: u8,
}

const fn init(x: usize) -> u8 {
    const TOP_KINDS: u8 = 2;
    if x == 0 {
        0 as u8
    } else {
        3 + TOP_KINDS * ((x - 1) as u8)
    }
}

impl Iterator for TakBoardIter {
    type Item = (ValidPiece, u8, u8); // PieceType, Square, Depth
    fn next(&mut self) -> Option<Self::Item> {
        // Will fix when Rust catches up
        const DEPTH_TABLE: [u8; 10] =
            [init(0), init(1), init(2), init(3), init(4), init(5), init(6), init(7), init(8), init(9)];
        if self.idx > self.board.data.data.len() {
            return None;
        }
        let val = self.board.data.data[self.idx];
        let square = val.square();
        if square >= 36 {
            return None;
        }
        let mut piece = val.piece();
        if square == self.last {
            self.depth += 1;
        } else {
            self.depth = 0;
            if self.board.data.caps[0] == square || self.board.data.caps[1] == square {
                piece = piece.promote_cap();
            }
        }
        self.idx += 1;
        self.last = square;
        Some((piece, square, DEPTH_TABLE[self.depth as usize]))
    }
}

#[derive(Clone)]
pub struct TakBinpackLoader<T: Fn(&ScoredPosition) -> bool> {
    file_path: [String; 1],
    buffer_size: usize,
    threads: usize,
    filter: T,
}

impl<T: Fn(&ScoredPosition) -> bool> TakBinpackLoader<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self {
            file_path: [path.to_string(); 1],
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ScoredPosition>() / 2,
            threads,
            filter,
        }
    }
}

impl<T> DataLoader<ScoredPosition> for TakBinpackLoader<T>
where
    T: Fn(&ScoredPosition) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_path
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[ScoredPosition]) -> bool>(&self, _: usize, batch_size: usize, mut f: F) {
        let file_path = self.file_path[0].clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();

        let reader_buffer_size = 16384 * threads;
        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<ScoredPosition>>(8);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);

            'dataloading: loop {
                let mut reader = CompressedTrainingDataEntryReader::new(&file_path).unwrap();
                loop {
                    while reader.has_next() {
                        let (next, full_size) = reader.next();
                        for val in next.into_iter() {
                            buffer.push(val);
                        }
                        reader.offset += full_size;
                    }
                    if buffer.len() == reader_buffer_size || !reader.has_next() {
                        if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(buffer).is_err() {
                            break 'dataloading;
                        }

                        buffer = Vec::with_capacity(reader_buffer_size);
                    }
                    if !reader.fetch_next_chunk_if_needed() {
                        break;
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Vec<ScoredPosition>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(unfiltered) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                thread::scope(|s| {
                    let chunk_size = unfiltered.len().div_ceil(threads);
                    let mut handles = Vec::new();

                    for chunk in unfiltered.chunks(chunk_size) {
                        let this_sender = converted_sender.clone();
                        let handle = s.spawn(move || {
                            let mut buffer = Vec::with_capacity(chunk_size);

                            for entry in chunk {
                                if filter(entry) {
                                    buffer.push(*entry);
                                }
                            }

                            this_sender.send(buffer).is_err()
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        if handle.join().unwrap() {
                            reader_msg_sender.send(true).unwrap();
                            should_break = true;
                        }
                    }
                });
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ScoredPosition>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle_and_rotate(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        let (batch_sender, batch_reciever) = mpsc::sync_channel::<Vec<ScoredPosition>>(16);
        let (batch_msg_sender, batch_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
                for batch in shuffle_buffer.chunks(batch_size) {
                    if batch_msg_receiver.try_recv().unwrap_or(false) || batch_sender.send(batch.to_vec()).is_err() {
                        buffer_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }
                }
            }
        });

        'dataloading: while let Ok(inputs) = batch_reciever.recv() {
            for batch in inputs.chunks(batch_size) {
                let should_break = f(batch);

                if should_break {
                    batch_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        }

        drop(batch_reciever);
    }
}

fn shuffle_and_rotate(data: &mut [ScoredPosition]) {
    let mut rng = SimpleRand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
    for val in data.iter_mut() {
        val.symmetry(rng.rng() as usize % 8);
    }
}

pub struct SimpleRand(u64);

impl SimpleRand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).expect("Guaranteed increasing.").as_micros() as u64
            & 0xFFFF_FFFF;

        Self(seed)
    }

    pub fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() {}
