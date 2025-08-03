use std::{
    fs::{File, OpenOptions},
    io::Read,
};

use anyhow::{Context, Result};
use bullet_lib::{
    default::{
        inputs::SparseInputType,
        loader::{self, LoadableDataType},
        outputs, Layout, QuantTarget, SavedFormat, Trainer,
    },
    loader::CanBeDirectlySequentiallyLoaded,
    lr, operations,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Node, Shape,
    TrainingSchedule, TrainingSteps,
};
use bulletformat::BulletFormat;

mod tak_utils;
use tak_utils::{BoardData, PieceSquare, TakSimple6, ValidPiece, HIDDEN_SIZE, NNUE6, QA, QB, SCALE};

use crate::tak_utils::ScoredPosition;

type GameData = ([u8; 2], [PieceSquare; 62], bool);
const TAKBOARD_SIZE: usize = std::mem::size_of::<BoardData>();

impl BulletFormat for ScoredPosition {
    type FeatureType = (ValidPiece, u8, u8);

    const HEADER_SIZE: usize = 0;

    fn set_result(&mut self, result: f32) {
        self.result = (2.0 * result) as u8;
    }

    fn score(&self) -> i16 {
        self.score.into()
    }

    fn result(&self) -> f32 {
        f32::from(self.result) / 2.
    }

    fn result_idx(&self) -> usize {
        usize::from(self.result)
    }
}

// impl LoadableDataType for ScoredPosition {
//     fn score(&self) -> i16 {
//         self.score
//     }

//     fn result(&self) -> loader::GameResult {
//         f32::from(self.result) / 2.
//     }
// }

impl SparseInputType for TakSimple6 {
    type RequiredDataType = ScoredPosition;

    fn num_inputs(&self) -> usize {
        Self::NUM_INPUTS
    }

    fn max_active(&self) -> usize {
        // Pieces, Side, Reserves
        62 + 1 + 2
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.handle_features(pos, f);
    }

    fn shorthand(&self) -> String {
        "Tak6x6Simple".to_string()
    }

    fn description(&self) -> String {
        "A simple NNUE representation of Tak 6x6".to_string()
    }
}

// impl FromStr for BoardData {
//     type Err = String;

//     fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
//         if let Ok((caps, data, white_to_move)) = interpret_tps(s) {
//             Ok(BoardData { caps, data, white_to_move, score: 0, result: 0 })
//         } else {
//             Err(format!("Could not parse: {}", s))
//         }
//     }
// }

// fn interpret_tps(full_tps: &str) -> Result<GameData> {
//     let (uncompressed_tps, rest) = full_tps.split_once(" ").context("Invalid TPS")?;
//     let mut idx = 0;
//     let mut caps = [255, 255];
//     let mut data = [PieceSquare(u8::MAX); 62];
//     let mut stack = Vec::new();
//     for (sq_idx, sq) in uncompressed_tps.split(&[',', '/']).enumerate() {
//         let bytes = sq.as_bytes();
//         for c in bytes {
//             match *c {
//                 b'1' => stack.push(PieceSquare::new(sq_idx, 0)),
//                 b'2' => stack.push(PieceSquare::new(sq_idx, 1)),
//                 b'S' => stack.last_mut().unwrap().promote_wall(),
//                 b'C' => {
//                     if caps[0] < 36 {
//                         // Is valid
//                         caps[1] = sq_idx as u8;
//                     } else {
//                         caps[0] = sq_idx as u8;
//                     }
//                 }
//                 b'x' => {
//                     break;
//                 }
//                 _ => unimplemented!(),
//             }
//         }
//         for val in stack.drain(..).rev().take(10) {
//             data[idx] = val;
//             idx += 1;
//         }
//     }
//     let white_to_move = b'1' == rest.as_bytes()[0];
//     Ok((caps, data, white_to_move))
// }

// fn convert_text(f_in: &str, f_out: &str) -> Result<()> {
//     const USE_WILEM: bool = false;
//     const SCORE_IS_RESULT: bool = false;
//     const REVERSE_RESULT: bool = false;
//     use std::io::BufRead;
//     let file = BufReader::new(OpenOptions::new().read(true).open(f_in)?);
//     let mut out_buf = Vec::new();
//     let mut output = BufWriter::new(OpenOptions::new().create(true).write(true).open(f_out)?);
//     let mut total = 0;
//     for line in file.lines().skip(1) {
//         let line = line?;
//         if line.starts_with("@") {
//             continue;
//         }
//         total += 1;
//         let split: Vec<_> = line.split(";").collect();
//         let (caps, data, white_to_move) = interpret_tps(split.get(0).context("No TPS")?)?;
//         let score = if USE_WILEM {
//             (split[7].parse::<f32>()? * SCALE as f32).trunc() as i16
//         } else if SCORE_IS_RESULT {
//             (split[2].parse::<i32>()? * SCALE) as i16
//         } else {
//             split[2].parse()?
//         };
//         let winner = split.get(3).unwrap_or(&"0");
//         let white_result = match *winner {
//             "1" => 2,
//             "0" => 1,
//             "-1" => 0,
//             _ => unimplemented!(),
//         };
//         let result = if REVERSE_RESULT {
//             if white_to_move {
//                 white_result
//             } else {
//                 2 - white_result
//             }
//         } else {
//             white_result
//         };
//         let board = BoardData { score, caps, data, white_to_move, result };
//         out_buf.push(board);
//         if out_buf.len() % 16384 == 0 {
//             BulletFormat::write_to_bin(&mut output, &out_buf).with_context(|| "Failed to write boards into output.")?;
//             out_buf.clear();
//         }
//     }
//     dbg!(total);
//     BulletFormat::write_to_bin(&mut output, &out_buf).with_context(|| "Failed to write boards into output.")?;

//     Ok(())
// }

// unsafe impl CanBeDirectlySequentiallyLoaded for BoardData {}

fn train(net_id: &str) -> Result<()> {
    let inputs = TakSimple6 {};
    let hl = HIDDEN_SIZE;
    let num_inputs = inputs.num_inputs();

    let (mut graph, output_node) = build_network(num_inputs, hl);

    graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (2.0 * hl as f32).sqrt(), true);
    graph.get_weights_mut("pst").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l1b").seed_random(0.0, 1.0 / (2.0 * hl as f32).sqrt(), true);

    let mut trainer = Trainer::<AdamWOptimiser, TakSimple6, outputs::Single>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::Single,
        vec![
            SavedFormat::new("l0w", QuantTarget::I16(QA), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(QA), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(QB), Layout::Normal),
            SavedFormat::new("pst", QuantTarget::I16(QA), Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::I16(QB * QA), Layout::Normal),
        ],
        false,
    );

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.35 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };
    // let files = &["out9.data", "out10.data"];
    // let mut files = Vec::new();
    // for f in std::fs::read_dir("/media/justin/SSD Ubuntu Stora/tak/bullet")? {
    //     let f = f?;
    //     if f.file_type()?.is_file() {
    //         let f = f.path();
    //         files.push(f.as_os_str().to_str().unwrap().to_string());
    //     }
    // }
    // let use_files: Vec<&str> = files.iter().map(|x| x.as_str()).collect();
    // let files = &["/media/justin/SSD Ubuntu Stora/tak/bullet.data"];
    // let data_loader = loader::DirectSequentialDataLoader::new(&use_files);
    // loading from a SF binpack
    let data_loader = {
        let file_path = "/home/justin/Code/rust/topaz-eval/data_total.bin";
        let buffer_size_mb = 1024;
        let threads = 4;
        fn filter(entry: &ScoredPosition) -> bool {
            entry.score.get().abs() <= 9999i16
        }

        tak_utils::TakBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);
    Ok(())
}

unsafe impl CanBeDirectlySequentiallyLoaded for ScoredPosition {}

fn build_network(inputs: usize, hl: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let nstm = builder.create_input("nstm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0w = builder.create_weights("l0w", Shape::new(hl, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(hl, 1));
    let l1w = builder.create_weights("l1w", Shape::new(1, hl * 2));
    let pst = builder.create_weights("pst", Shape::new(1, inputs));
    let l1b = builder.create_weights("l1b", Shape::new(1, 1));

    // inference
    let l1 = operations::sparse_affine_dual_with_activation(&mut builder, l0w, stm, nstm, l0b, Activation::SCReLU);
    let l2 = operations::affine(&mut builder, l1w, l1, l1b);
    let psqt = operations::matmul(&mut builder, pst, stm);
    let predicted = operations::add(&mut builder, l2, psqt);

    let sigmoided = operations::activate(&mut builder, predicted, Activation::Sigmoid);
    operations::mse(&mut builder, sigmoided, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}

fn main() {
    let train_net = true;
    let net_id = "7_28_25";
    if train_net {
        train(net_id).unwrap();
    }
    dbg!("Hello World");
}
