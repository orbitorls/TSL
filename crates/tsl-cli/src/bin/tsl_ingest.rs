//! `tsl-ingest` — fingerspelling zip/tree ingest and TSL-51 HF metadata.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tsl_ingest::{
    ingest_fingerspelling, ingest_tsl51_metadata, FingerspellingIngestOptions, Tsl51IngestOptions,
};

#[derive(Parser, Debug)]
#[command(
    name = "tsl-ingest",
    about = "Prepare One-Stage-TFS and TSL-51 datasets locally"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Extract or validate One-Stage-TFS fingerspelling layout.
    Fingerspelling {
        #[arg(long)]
        zip: Option<PathBuf>,
        #[arg(long)]
        dataset_root: Option<PathBuf>,
        #[arg(long, default_value = "data/one_stage_tfs")]
        output: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Copy or download TSL-51 metadata CSVs.
    Tsl51 {
        #[arg(long)]
        source: Option<PathBuf>,
        #[arg(long, default_value = "data/tsl51_raw")]
        output: PathBuf,
        #[arg(long)]
        force: bool,
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Fingerspelling {
            zip,
            dataset_root,
            output,
            force,
        } => {
            let manifest = ingest_fingerspelling(FingerspellingIngestOptions {
                zip_path: zip,
                dataset_root,
                output,
                force,
            })?;
            println!(
                "[ok] fingerspelling: {} classes, {} train images",
                manifest.num_classes, manifest.num_training_images
            );
        }
        Command::Tsl51 {
            source,
            output,
            force,
        } => {
            let manifest = ingest_tsl51_metadata(Tsl51IngestOptions {
                source,
                output,
                force,
            })?;
            println!(
                "[ok] tsl51: {} classes, {} training rows",
                manifest.num_classes, manifest.num_training_rows
            );
        }
    }
    Ok(())
}
