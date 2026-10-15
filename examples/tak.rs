use anyhow::Result;
use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{optimiser::AdamW, InitSettings},
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::CanBeDirectlySequentiallyLoaded, ValueTrainerBuilder},
    Shape,
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

#[derive(Clone, Copy, Default)]
pub struct SideBuckets;
impl OutputBuckets<ScoredPosition> for SideBuckets {
    const BUCKETS: usize = 2;
    fn bucket(&self, pos: &ScoredPosition) -> u8 {
        if pos.data.white_to_move {
            0
        } else {
            1
        }
    }
}

// // This is stupid and unsound because of how I did the compression.
// #[derive(Clone, Copy, Default)]
// pub struct CapstoneOutputBuckets;
// impl OutputBuckets<ScoredPosition> for CapstoneOutputBuckets {
//     const BUCKETS: usize = 37;

//     fn bucket(&self, pos: &ScoredPosition) -> u8 {
//         if pos.data.white_to_move {
//             std::cmp::min(Self::BUCKETS as u8 - 1, pos.data.caps[0])
//         } else {
//             std::cmp::min(Self::BUCKETS as u8 - 1, pos.data.caps[1])
//         }
//         36
//     }
// }

#[derive(Clone, Copy, Default)]
pub struct CountBuckets<const N: usize>;
impl<const N: usize> OutputBuckets<ScoredPosition> for CountBuckets<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ScoredPosition) -> u8 {
        let num_pieces = pos.data.data_len;
        let divisor = 64usize.div_ceil(N); // 62
                                           // (pos.occ().count_ones() as u8 - 2) / divisor as u8
        num_pieces / divisor as u8
    }
}
fn train(net_id: &str) -> Result<()> {
    // const NUM_OUTPUT_BUCKETS: usize = 8;
    let inputs = TakSimple6 {};
    let hl = HIDDEN_SIZE;
    let num_inputs = inputs.num_inputs();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(TakSimple6 {})
        .output_buckets(SideBuckets)
        .save_format(&[
            SavedFormat::id("l0w").quantise::<i16>(QA),
            SavedFormat::id("l0b").quantise::<i16>(QA),
            SavedFormat::id("l1w").quantise::<i16>(QB).transpose(),
            SavedFormat::id("pst").quantise::<i16>(QA),
            SavedFormat::id("l1b").quantise::<i16>(QB * QA),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // weights
            let l0 = builder.new_affine("l0", num_inputs, hl);
            let l1 = builder.new_affine("l1", 2 * hl, SideBuckets::BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            let out = l1.forward(hidden_layer).select(output_buckets);
            let psqt = builder.new_weights(
                "pst",
                Shape::new(1, num_inputs),
                InitSettings::Normal { mean: 0.0, stdev: 1.0 / (num_inputs as f32).sqrt() },
            );
            out + psqt.matmul(stm_inputs)
        });

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
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.00001, final_superbatch: 240 },
        // lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = {
        let file_path = "/media/justin/SSD Ubuntu Stora/tak/2025_nets/data/data.bin";
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

fn main() {
    let train_net = true;
    let net_id = "8_17_26";
    if train_net {
        train(net_id).unwrap();
    }
    dbg!("Hello World");
}
