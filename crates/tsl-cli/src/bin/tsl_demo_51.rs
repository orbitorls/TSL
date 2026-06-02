//! TSL-51 word demo — mirrors `webcam_word_demo.py` CLI.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tsl_core::sequence::SequenceBuffer;
use tsl_infer::{load_demo_artifacts, DemoTrack, EmaBuffer};

#[derive(Parser, Debug)]
#[command(name = "tsl-demo-51", about = "TSL-51 word-level demo (Rust)")]
struct Args {
    #[arg(long, default_value = "tsl51_model.keras")]
    model: PathBuf,
    #[arg(long, default_value = "tsl51_labels.json")]
    labels: PathBuf,
    #[arg(long, default_value = "tsl51_scaler.json")]
    scaler: PathBuf,
    #[arg(long, default_value_t = 0)]
    cam: u32,
    #[arg(long, default_value_t = 0.55)]
    threshold: f32,
    #[arg(long, default_value_t = 0.35)]
    smoothing_alpha: f32,
    #[arg(long, default_value_t = 2)]
    frame_skip: u32,
    #[arg(long)]
    dry_run: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _ = (args.threshold, args.frame_skip);

    if args.dry_run {
        let mut buf = SequenceBuffer::new(60);
        for v in 1u8..=3 {
            buf.push([v as f32; 162]);
        }
        println!("dry-run OK: buffer_len={}", buf.len());
        return Ok(());
    }

    let artifacts = load_demo_artifacts(
        &args.model,
        &args.labels,
        &args.scaler,
        DemoTrack::Tsl51,
        None,
        Some("Train TSL-51; export tsl51_model.onnx + tsl51_scaler.json"),
    )
    .context("load tsl51 artifacts")?;

    let mut buf = SequenceBuffer::new(60);
    let mut ema = EmaBuffer::new(args.smoothing_alpha);
    for v in 1u8..=3 {
        buf.push([v as f32; 162]);
    }
    let seq_flat: Vec<f32> = buf.get_padded_flat();
    let scaled: Vec<f32> = seq_flat
        .chunks(162)
        .flat_map(|row| artifacts.scaler.transform(row).unwrap())
        .collect();
    let probs = artifacts.predictor.predict(&scaled)?;
    let smoothed = ema.update(&probs)?;
    println!(
        "artifact smoke OK: top_prob={:.4}",
        smoothed.iter().cloned().fold(0.0f32, f32::max)
    );
    eprintln!("Webcam not wired. cam={}", args.cam);
    Ok(())
}
