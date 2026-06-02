//! Fingerspelling webcam demo (Rust) — mirrors `webcam_demo.py` CLI args.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tsl_infer::{load_demo_artifacts, DemoTrack, EmaBuffer};
use tsl_vision::{FrameInput, InjectedStubBackend, LandmarkBackend, OnnxModelPaths};

#[derive(Parser, Debug)]
#[command(name = "tsl-demo-fs", about = "Thai fingerspelling live demo (Rust)")]
struct Args {
    #[arg(long, default_value = "model.keras")]
    model: PathBuf,
    #[arg(long, default_value = "labels.json")]
    labels: PathBuf,
    #[arg(long, default_value = "scaler.json")]
    scaler: PathBuf,
    #[arg(long, default_value_t = 0)]
    cam: u32,
    #[arg(long, default_value_t = 0.7)]
    threshold: f32,
    #[arg(long, default_value_t = 0.4)]
    smoothing_alpha: f32,
    #[arg(long, default_value_t = 1)]
    frame_skip: u32,
    #[arg(long, default_value_t = 2)]
    top_k: usize,
    #[arg(long)]
    save_log: Option<PathBuf>,
    #[arg(long)]
    use_tflite: bool,
    /// Run without ONNX artifacts using injected stub landmarks.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _ = (args.use_tflite, args.threshold, args.top_k);

    // Use non-degenerate synthetic landmarks so wrist->midpoint span is non-zero.
    let mut coords = [0.0f32; 63];
    for i in 0..21 {
        let base = i * 3;
        coords[base] = i as f32 * 0.01;
        coords[base + 1] = i as f32 * 0.015;
        coords[base + 2] = i as f32 * 0.005;
    }
    let mut backend = InjectedStubBackend::from_hand_coords(coords);
    let frame_in = FrameInput::new(1, 1, &[0u8; 3]);
    let hand = backend
        .detect_hand(&frame_in)?
        .ok_or_else(|| anyhow::anyhow!("stub hand frame (normalization failed?)"))?;

    if args.dry_run {
        println!("dry-run OK: hand_dim={}", hand.landmarks.len());
        let paths = OnnxModelPaths::from_repo_root(".");
        eprintln!(
            "vision ONNX paths: hand={} (exists={})",
            paths.hand_landmarker.display(),
            paths.hand_landmarker.is_file()
        );
        return Ok(());
    }

    let artifacts = load_demo_artifacts(
        &args.model,
        &args.labels,
        &args.scaler,
        DemoTrack::Fingerspelling,
        None,
        Some("Train in Colab; export ONNX + scaler.json (scripts/migrate_scaler.py)"),
    )
    .context("load demo artifacts")?;

    let mut ema = EmaBuffer::new(args.smoothing_alpha.clamp(0.05, 1.0));
    let scaled = artifacts.scaler.transform(hand.as_slice())?;
    let probs = artifacts.predictor.predict(&scaled)?;
    let smoothed = ema.update(&probs)?;
    println!(
        "artifact smoke OK: classes={} top_prob={:.4}",
        artifacts.labels.len(),
        smoothed.iter().cloned().fold(0.0f32, f32::max)
    );

    eprintln!(
        "Webcam capture not wired. cam={} frame_skip={}",
        args.cam, args.frame_skip
    );
    if let Some(log) = args.save_log {
        eprintln!("  save_log={}", log.display());
    }
    Ok(())
}
