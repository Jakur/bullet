use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter},
    str::FromStr,
};

use bullet_lib::{
    default::{inputs::SparseInputType, loader, outputs, Layout, QuantTarget, SavedFormat, Trainer},
    loader::CanBeDirectlySequentiallyLoaded,
    lr, operations,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Node, Shape,
    TrainingSchedule, TrainingSteps,
};

mod tak_utils;
use tak_utils::{
    Accumulator, Incremental, IncrementalState, Network, PieceSquare, TakBoard, TakSimple6, ValidPiece, HIDDEN_SIZE,
    QA, QB, SCALE,
};

use anyhow::{Context, Result};
use bulletformat::BulletFormat;

type GameData = ([u8; 2], [PieceSquare; 62], bool);

impl BulletFormat for TakBoard {
    type FeatureType = (ValidPiece, u8, u8);

    const HEADER_SIZE: usize = 0;

    fn set_result(&mut self, result: f32) {
        self.result = (2.0 * result) as u8;
    }

    fn score(&self) -> i16 {
        self.score
    }

    fn result(&self) -> f32 {
        f32::from(self.result) / 2.
    }

    fn result_idx(&self) -> usize {
        usize::from(self.result)
    }
}

impl SparseInputType for TakSimple6 {
    type RequiredDataType = TakBoard;

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

impl FromStr for TakBoard {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if let Ok((caps, data, white_to_move)) = interpret_tps(s) {
            Ok(TakBoard { caps, data, white_to_move, score: 0, result: 0 })
        } else {
            Err(format!("Could not parse: {}", s))
        }
    }
}

fn interpret_tps(full_tps: &str) -> Result<GameData> {
    let (uncompressed_tps, rest) = full_tps.split_once(" ").context("Invalid TPS")?;
    let mut idx = 0;
    let mut caps = [255, 255];
    let mut data = [PieceSquare(u8::MAX); 62];
    let mut stack = Vec::new();
    for (sq_idx, sq) in uncompressed_tps.split(&[',', '/']).enumerate() {
        let bytes = sq.as_bytes();
        for c in bytes {
            match *c {
                b'1' => stack.push(PieceSquare::new(sq_idx, 0)),
                b'2' => stack.push(PieceSquare::new(sq_idx, 1)),
                b'S' => stack.last_mut().unwrap().promote_wall(),
                b'C' => {
                    if caps[0] < 36 {
                        // Is valid
                        caps[1] = sq_idx as u8;
                    } else {
                        caps[0] = sq_idx as u8;
                    }
                }
                b'x' => {
                    break;
                }
                _ => unimplemented!(),
            }
        }
        for val in stack.drain(..).rev().take(10) {
            data[idx] = val;
            idx += 1;
        }
    }
    let white_to_move = b'1' == rest.as_bytes()[0];
    Ok((caps, data, white_to_move))
}

fn convert_text(f_in: &str, f_out: &str) -> Result<()> {
    use std::io::BufRead;
    let file = BufReader::new(OpenOptions::new().read(true).open(f_in)?);
    let mut out_buf = Vec::new();
    let mut output = BufWriter::new(OpenOptions::new().create(true).write(true).open(f_out)?);
    for line in file.lines().skip(1) {
        let line = line?;
        if line.starts_with("@") {
            continue;
        }
        let split: Vec<_> = line.split(";").collect();
        let (caps, data, white_to_move) = interpret_tps(split.get(1).context("No TPS")?)?;
        let score = split[2].parse()?;
        let winner = split[6];
        let white_result = match winner {
            "1" => 2,
            "0" => 1,
            "-1" => 0,
            _ => unimplemented!(),
        };
        let result = if white_to_move { white_result } else { 2 - white_result };
        let board = TakBoard { score, caps, data, white_to_move, result };
        out_buf.push(board);
        if out_buf.len() % 16384 == 0 {
            BulletFormat::write_to_bin(&mut output, &out_buf).with_context(|| "Failed to write boards into output.")?;
            out_buf.clear();
        }
    }
    BulletFormat::write_to_bin(&mut output, &out_buf).with_context(|| "Failed to write boards into output.")?;

    Ok(())
}

unsafe impl CanBeDirectlySequentiallyLoaded for TakBoard {}

fn train() -> Result<()> {
    let inputs = TakSimple6 {};
    let hl = HIDDEN_SIZE;
    let num_inputs = inputs.num_inputs();

    let (mut graph, output_node) = build_network(num_inputs, hl);

    graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (2.0 * hl as f32).sqrt(), true);
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
            SavedFormat::new("l1b", QuantTarget::I16(QB * QA), Layout::Normal),
            SavedFormat::new("pst", QuantTarget::I16(QA), Layout::Normal),
        ],
        false,
    );

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["out9.data", "out10.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    // let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    // println!("Eval: {eval:.3}cp");
    Ok(())
}

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
    let l1b = builder.create_weights("l1b", Shape::new(1, 1));
    let pst = builder.create_weights("pst", Shape::new(1, inputs));

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

fn sanity_check() -> Result<()> {
    let inputs = TakSimple6 {};
    let num_inputs = inputs.num_inputs();

    let (graph, output_node) = build_network(num_inputs, HIDDEN_SIZE);

    let mut trainer = Trainer::<AdamWOptimiser, TakSimple6, outputs::Single>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::Single,
        vec![
            SavedFormat::new("l0w", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(64), Layout::Normal),
            SavedFormat::new("pst", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::I16(64 * 255), Layout::Normal),
        ],
        false,
    );
    trainer.load_from_checkpoint("checkpoints/test-240");
    // trainer.save_to_checkpoint("checkpoints/test-240a");
    // dbg!(trainer.optimiser().graph().get_weights("l0w").debug_reduce());
    // let tps1 = "2,x,x,x,1,1/x,x,2,2,2,12/x,x,2,1,1,1/x,x,2,1C,x,x/x,x,2C,1,x,x/x,x,x,1,x,x 1 10";
    let tps1 = "2,1,x,x,x,x/x,2,1C,2,2,x/x,1,2,2C,1,2/x,1,1,2,1,x/x,x,x,1,x,x/x,x,x,x,x,1 2 9";
    let score = trainer.eval(tps1);
    dbg!(score);
    let (score_manual, ours, theirs) = manual_eval(tps1);
    // let tps2 = "2,x3,1,x/x2,2,2,2,121/x2,2,1,1,1/x2,2,1C,x2/x2,2C,1,x2/x3,1,x2 2 10";
    // let tps2 = "2,1,x,x,x,x/x,2,1C,2,2,x/x,1,2,2C,1,2/x,1,1,2,1,x/x,x,x,1,x,x/x,x,x,x,x,1 2 9";
    let tps2 = "2,1,x,x,x,x/x,2,1C,2,2,x/x,1,2,2C,1,2/x,1,1,2,1,x/x,x,1,x,x,x/x,x,x,x,x,1 2 9";
    dbg!(score_manual);
    let score2 = trainer.eval(tps2);
    let (score_manual2, _, _) = manual_eval(tps2);
    dbg!(score2);
    dbg!(score_manual2);

    let (score_incremental2, _, _) = incremental_eval(tps2, &ours, &theirs);
    dbg!(score_incremental2);
    Ok(())
}

static NNUE: Network = unsafe {
    let bytes = include_bytes!("../checkpoints/test-240a/quantised.bin");
    assert!(bytes.len() == std::mem::size_of::<Network>());
    std::mem::transmute(*bytes)
};

fn build_features(takboard: TakBoard) -> (IncrementalState, IncrementalState) {
    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    let simple = TakSimple6 {};
    simple.map_features(&takboard, |x, y| {
        ours.push(x as u16);
        theirs.push(y as u16);
    });
    (IncrementalState::from_vec(ours), IncrementalState::from_vec(theirs))
}

fn incremental_eval(tps: &str, old_ours: &Incremental, old_theirs: &Incremental) -> (i32, Incremental, Incremental) {
    let tb = TakBoard::from_str(tps).unwrap();
    let (ours, theirs) = build_features(tb);
    // Ours
    let mut ours_acc = Accumulator::from_old(&old_ours.vec);
    let (sub, add) = ours.diff(&old_ours.state);
    ours_acc.remove_all(&sub, &NNUE);
    ours_acc.add_all(&add, &NNUE);
    let ours = Incremental { state: ours, vec: ours_acc };
    // Theirs
    let mut theirs_acc = Accumulator::from_old(&old_theirs.vec);
    let (sub, add) = theirs.diff(&old_theirs.state);
    theirs_acc.remove_all(&sub, &NNUE);
    theirs_acc.add_all(&add, &NNUE);
    let theirs = Incremental { state: theirs, vec: theirs_acc };
    // Output
    let eval = NNUE.evaluate(&ours.vec, &theirs.vec, &ours.state.piece_data, &ours.state.meta);
    (eval, ours, theirs)
}

fn manual_eval(tps: &str) -> (i32, Incremental, Incremental) {
    let tb = TakBoard::from_str(tps).unwrap();
    let (ours, theirs) = build_features(tb);
    let ours = Incremental::fresh_new(&NNUE, ours);
    let theirs = Incremental::fresh_new(&NNUE, theirs);
    let eval = NNUE.evaluate(&ours.vec, &theirs.vec, &ours.state.piece_data, &ours.state.meta);

    (eval, ours, theirs)
}

fn main() {
    let convert = false;
    let train_net = false;
    let check = true;
    if convert {
        convert_text("/home/justin/Code/rust/topaz-eval/games9_new.csv", "out9.data").unwrap();
    }
    if train_net {
        train().unwrap();
    }
    if check {
        sanity_check().unwrap();
    }
    dbg!("Hello World");
}
