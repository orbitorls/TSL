#!/usr/bin/env python3
"""
TSL TUI - Thai Sign Language Recognition System
Complete TUI with all functions and configurable parameters
"""
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, Header, Footer, Static, Input,
    Label, Select, Log
)
from textual import work
from textual.worker import Worker

# Module logger
logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND DEFINITIONS - ALL FUNCTIONS WITH CONFIGURABLE PARAMS
# ============================================================================
COMMANDS: Dict[str, Dict] = {
    # TRAINING
    "train_gru": {
        "name": "Train GRU",
        "desc": "Train GRU model with K-Fold Cross Validation",
        "cmd": "python train_tsl51_v3.py",
        "params": [
            ("Model", "model", "gru", "select", [("GRU", "gru"), ("MLP", "mlp")]),
            ("Epochs", "epochs", "50", "input", None),
            ("Batch Size", "batch", "128", "input", None),
            ("Layers", "layers", "3", "input", None),
            ("Hidden Dim", "hidden", "256", "input", None),
            ("Learning Rate", "lr", "0.001", "input", None),
            ("Augmentation", "augment", "5", "input", None),
            ("K-Folds", "k_folds", "0", "input", None),
        ]
    },
    "train_mlp": {
        "name": "Train MLP",
        "desc": "Train MLP model (faster, simpler)",
        "cmd": "python train_tsl51_v3.py --model mlp",
        "params": [
            ("Epochs", "epochs", "50", "input", None),
            ("Batch Size", "batch", "128", "input", None),
            ("Layers", "layers", "3", "input", None),
            ("Hidden Dim", "hidden", "256", "input", None),
            ("Learning Rate", "lr", "0.001", "input", None),
            ("Augmentation", "augment", "5", "input", None),
        ]
    },
    "train_expert": {
        "name": "Train Expert",
        "desc": "Train Expert model variant",
        "cmd": "python train_tsl51_v3.py --dataset tsl51_expert",
        "params": [
            ("Epochs", "epochs", "30", "input", None),
            ("Batch Size", "batch", "64", "input", None),
        ]
    },
    
    # INFERENCE
    "inference": {
        "name": "Inference",
        "desc": "Run inference on landmark data (.npz)",
        "cmd": "python inference.py --model models/tsl51_gru_best.pt --input data.npz --top-k 3",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Input File", "input", "data.npz", "input", None),
            ("Top K", "top_k", "3", "input", None),
        ]
    },
    "predict_video": {
        "name": "Predict Video",
        "desc": "Predict from video file using MediaPipe",
        "cmd": "python predict_video.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Video Path", "input", "video.mp4", "input", None),
        ]
    },
    "camera": {
        "name": "Camera Translation",
        "desc": "Real-time camera translation",
        "cmd": "python camera_translate.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Camera Index", "camera", "0", "input", None),
        ]
    },
    "translate": {
        "name": "Translate JSON",
        "desc": "Translate JSON landmark files",
        "cmd": "python translate.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Input JSON", "input", "landmarks.json", "input", None),
        ]
    },
    
    # DATA
    "download_tsl51": {
        "name": "Download TSL-51",
        "desc": "Download TSL-51 dataset from HuggingFace",
        "cmd": "python scripts/archive/data/download_tsl51_v2.py",
        "params": [
            ("Max Samples", "max_samples", "", "input", None),
        ]
    },
    "download_expert": {
        "name": "Download Expert",
        "desc": "Download Expert dataset",
        "cmd": "python scripts/archive/data/download_expert_full.py",
        "params": [
            ("Max Samples", "max_samples", "", "input", None),
        ]
    },
    "clear_cache": {
        "name": "Clear Cache",
        "desc": "Clear cached datasets",
        "cmd": 'python -c "import shutil; shutil.rmtree(\'.cache\', ignore_errors=True); print(\'Cache cleared\')"',
        "params": []
    },
    
    # EXPORT
    "export_model": {
        "name": "Export Model",
        "desc": "Export trained model to portable format",
        "cmd": "python root_archive/export_model.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
        ]
    },
    "export_onnx": {
        "name": "Export ONNX",
        "desc": "Export model to ONNX format",
        "cmd": "python root_archive/export_onnx.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
        ]
    },
    
    # ANALYSIS
    "benchmark": {
        "name": "Benchmark",
        "desc": "Run video benchmark (WER/BLEU/ROUGE)",
        "cmd": "python benchmark_models.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Max Videos", "max", "50", "input", None),
        ]
    },
    "reports": {
        "name": "Reports",
        "desc": "Create benchmark reports",
        "cmd": "python root_archive/create_benchmark_report.py",
        "params": []
    },
    "check_data": {
        "name": "Check Data",
        "desc": "Run data verification scripts",
        "cmd": "python -c \"import os; [os.system(f'python scripts/archive/verify/{f}') for f in os.listdir('scripts/archive/verify') if f.endswith('.py')]\"",
        "params": []
    },
    
    # WEBSITE
    "website_dev": {
        "name": "Website Dev",
        "desc": "Start TSL website (localhost)",
        "cmd": "cd tsl-website && npm run dev",
        "params": []
    },
    "website_deploy": {
        "name": "Website Deploy",
        "desc": "Deploy to Vercel",
        "cmd": "npx vercel --dir tsl-website",
        "params": []
    },
}


# TRAINING PRESETS
PRESETS = {
    "quick": {"model": "gru", "epochs": "5", "batch": "64", "layers": "2", "hidden": "128", "augment": "0", "k_folds": "5"},
    "default": {"model": "gru", "epochs": "50", "batch": "128", "layers": "3", "hidden": "256", "augment": "5", "k_folds": "5"},
    "full_cv": {"model": "gru", "epochs": "100", "batch": "128", "layers": "3", "hidden": "256", "augment": "10", "k_folds": "5"},
    "mlp": {"model": "mlp", "epochs": "50", "batch": "128", "layers": "3", "hidden": "256", "augment": "5", "k_folds": "5"},
    "large_dataset": {"model": "gru", "epochs": "100", "batch": "256", "layers": "4", "hidden": "512", "augment": "0", "k_folds": "5"},
}
TRAINING_PRESETS = PRESETS  # Alias for compatibility


# ============================================================================
# MAIN APP
# ============================================================================
class TSLApp(App):
    """TSL TUI - Complete Thai Sign Language Recognition Interface."""
    
    CSS = """
    Screen {
        background: $primary-darken-3;
    }
    
    /* Right Panel */
    #right-panel {
        width: 1fr;
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    /* Left Panel */
    #left-panel {
        width: 40%;
        height: 100%;
        background: $surface-darken-1;
    }
    
    /* Section Headers */
    .section-header {
        text-style: bold;
        margin: 1 0 0 0;
        padding: 0 1;
        color: $accent;
    }
    
    /* Training Presets */
    #presets-container {
        height: auto;
        padding: 1;
        background: $surface-darken-2;
        border: solid $border;
        margin: 0 1 1 1;
    }
    
    .preset-btn {
        width: 100%;
        margin: 0 0 1 0;
        border: none;
    }
    
    /* Config Inputs */
    #config-container {
        height: auto;
        padding: 1;
        background: $surface-darken-2;
        border: solid $border;
        margin: 0 1 1 1;
    }
    
    Input.config-input {
        width: 100%;
        margin: 0 0 1 0;
        border: solid $border;
        background: $surface;
    }
    
    /* Quick Actions */
    #quick-actions {
        height: auto;
        padding: 1;
        layout: horizontal;
    }
    
    /* Output Area */
    #output-container {
        height: 1fr;
        padding: 1;
    }
    
    Log#output-log {
        height: 100%;
        border: solid $border;
        background: $surface-darken-3;
    }
    
    /* Status */
    #status-bar {
        height: 3;
        padding: 0 1;
        background: $primary;
        color: $text;
        text-style: bold;
    }
    
    /* Buttons */
    Button {
        width: 100%;
        border: none;
    }
    
    .action-btn {
        width: 50%;
    }
    
    /* Running state */
    .running-indicator {
        color: $warning;
        text-style: bold;
    }
    
    /* Dataset Selection */
    #dataset-container {
        height: auto;
        padding: 1;
        background: $surface-darken-2;
        border: solid $border;
        margin: 0 1 1 1;
    }
    
    /* Navigation */
    #nav-container {
        height: auto;
        padding: 1;
        background: $surface-darken-2;
        border: solid $border;
        margin: 0 1 1 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "preset_quick", "Quick"),
        Binding("2", "preset_default", "Default"),
        Binding("3", "preset_full", "Full CV"),
        Binding("4", "preset_large", "Large Dataset"),
        Binding("r", "run_training", "Run"),
        Binding("s", "stop_training", "Stop"),
        Binding("c", "clear_output", "Clear"),
        Binding("d", "show_dashboard", "Dashboard"),
        Binding("v", "run_data_validation", "Data Validation"),
        Binding("b", "run_model_benchmark", "Model Benchmark"),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_preset = "default"
        self.current_cmd_id = "train_gru"  # Currently selected command
        self.worker: Optional[Worker] = None
        self.training_active = False
        # Track the running subprocess so we can terminate it on stop
        self._current_process: Optional[asyncio.subprocess.Process] = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal():
            # ===== LEFT PANEL =====
            with VerticalScroll(id="left-panel"):
                # Title
                yield Static("[bold cyan]TSL TUI[/bold cyan] [dim]v2.0[/dim]", classes="section-header")
                yield Static("[dim]Thai Sign Language Recognition[/dim]\n", classes="section-header")
                
                # Training Presets
                yield Static("[yellow]PRESETS[/yellow]", classes="section-header")
                with Vertical(id="presets-container"):
                    yield Button("1. Quick (5 epochs)", id="btn-preset-quick", classes="preset-btn", variant="primary")
                    yield Button("2. Default (50 epochs)", id="btn-preset-default", classes="preset-btn", variant="success")
                    yield Button("3. Full K-Fold CV", id="btn-preset-full", classes="preset-btn", variant="success")
                    yield Button("4. MLP Fast", id="btn-preset-mlp", classes="preset-btn", variant="warning")
                    yield Button("5. Large Dataset (~45k)", id="btn-preset-large", classes="preset-btn", variant="primary")
                
                # Dataset Selection
                yield Static("[magenta]DATASET[/magenta]", classes="section-header")
                with Vertical(id="dataset-container"):
                    yield Label("Select Dataset:")
                    yield Select([
                        ("User Sign (547 samples)", "tsl51_user_sign"),
                        ("Expert (1,155 samples)", "tsl51_expert"),
                        ("Expert Full (~45k samples)", "tsl51_expert_full"),
                        ("Combined (1,702 samples)", "tsl51_combined"),
                    ], value="tsl51_user_sign", id="sel-dataset")
                
                # Configuration
                yield Static("[cyan]CONFIGURATION[/cyan]", classes="section-header")
                with Vertical(id="config-container"):
                    yield Label("Model:")
                    yield Select([("GRU", "gru"), ("MLP", "mlp"), ("MOPGRU", "mopgru"), ("Hybrid", "hybrid")], value="gru", id="sel-model")
                    
                    yield Label("Epochs:")
                    yield Input(value="50", id="inp-epochs", classes="config-input")
                    
                    yield Label("Batch Size:")
                    yield Input(value="128", id="inp-batch", classes="config-input")
                    
                    yield Label("Layers:")
                    yield Input(value="3", id="inp-layers", classes="config-input")
                    
                    yield Label("Hidden Dim:")
                    yield Input(value="256", id="inp-hidden", classes="config-input")
                    
                    yield Label("Learning Rate:")
                    yield Input(value="0.001", id="inp-lr", classes="config-input")
                    
                    yield Label("Augmentation (0=off):")
                    yield Input(value="5", id="inp-augment", classes="config-input")
                    
                    yield Label("K-Folds (blank=training default):")
                    yield Input(value="5", id="inp-kfolds", classes="config-input")
                
                # Action Buttons
                yield Static("[green]ACTIONS[/green]", classes="section-header")
                with Horizontal(id="quick-actions"):
                    yield Button("RUN", id="btn-run", variant="success", classes="action-btn")
                    yield Button("STOP", id="btn-stop", variant="error", classes="action-btn", disabled=True)
                
                # Navigation
                yield Static("[blue]NAVIGATION[/blue]", classes="section-header")
                with Vertical(id="nav-container"):
                    yield Button("Dashboard", id="btn-dashboard", variant="primary")
                    yield Button("Data Validation", id="btn-data-validation")
                    yield Button("Model Benchmark", id="btn-model-benchmark")
                
                # Other Commands (restored minimal set)
                yield Static("[blue]ACTIONS[/blue]", classes="section-header")
                yield Button("Inference", id="btn-inference")
                yield Button("Camera", id="btn-camera")
                yield Button("Download TSL-51", id="btn-download_tsl51")
                yield Button("Download Expert", id="btn-download_expert")
                yield Button("Clear Output", id="btn-clear")
            
            # ===== RIGHT PANEL =====
            with Vertical(id="right-panel"):
                yield Static("[bold]Console Output[/bold]", classes="section-header")
                with Vertical(id="output-container"):
                    yield Log(id="output-log", auto_scroll=True, highlight=True)
                yield Static("Ready", id="status-bar")
        
        yield Footer()
    
    def on_mount(self) -> None:
        self.title = "TSL TUI - Thai Sign Language Recognition"
        self.sub_title = "[b]q[/b]=Quit | [b]r[/b]=Run | [b]s[/b]=Stop | [b]1-4[/b]=Presets | [b]d[/b]=Dashboard | [b]v[/b]=Validate | [b]b[/b]=Benchmark"
        self._log("[cyan]TSL TUI initialized. Select a preset or configure manually.[/cyan]")
        self._load_preset("default")
    
    def _log(self, msg: str) -> None:
        """Write to output log."""
        try:
            log = self.query_one("#output-log", Log)
            log.write(msg)
        except Exception as e:
            # UI log may not be available (headless or during startup). Fallback to stderr.
            logger.debug("UI log not available, falling back to stderr: %s", e, exc_info=True)
            try:
                sys.stderr.write(msg + "\n")
            except Exception as e2:
                # Avoid silent failures; log full exception
                logger.exception("Failed to write log message to stderr: %s", e2)
    
    def _update_status(self, msg: str) -> None:
        """Update status bar."""
        try:
            status = self.query_one("#status-bar", Static)
            status.update(msg)
        except Exception as e:
            # Non-fatal: status widget may not be mounted yet
            logger.debug("Failed to update status bar (%s): %s", msg, e, exc_info=True)
    
    def _load_preset(self, preset_name: str) -> None:
        """Load a training preset into the config inputs."""
        preset = TRAINING_PRESETS.get(preset_name, {"epochs": "50", "batch": "128", "layers": "3", "hidden": "256", "augment": "5", "k_folds": "0"})
        self.current_preset = preset_name
        self.current_cmd_id = "train_gru"  # Default to training
        
        # Update inputs if they exist in the UI
        try:
            if "epochs" in preset:
                self.query_one("#inp-epochs", Input).value = str(preset.get("epochs", "50"))
            if "batch" in preset:
                self.query_one("#inp-batch", Input).value = str(preset.get("batch", "128"))
            if "layers" in preset:
                self.query_one("#inp-layers", Input).value = str(preset.get("layers", "3"))
            if "hidden" in preset:
                self.query_one("#inp-hidden", Input).value = str(preset.get("hidden", "256"))
            if "augment" in preset:
                self.query_one("#inp-augment", Input).value = str(preset.get("augment", "5"))
            if "k_folds" in preset:
                self.query_one("#inp-kfolds", Input).value = str(preset.get("k_folds", "0"))
            self._log(f"[green]Preset '{preset_name}' loaded[/green]")
        except Exception:
            # Partial preset load; still usable but surface the issue for debugging
            self._log(f"[yellow]Preset '{preset_name}' loaded (partial)[/yellow]")
            # Log exception with context
            logger.exception("_load_preset partial failure for %s", preset_name)
    
    def _get_training_command(self) -> str:
        """Build training command from current config."""
        try:
            dataset = self.query_one("#sel-dataset", Select).value
            model = self.query_one("#sel-model", Select).value
            epochs = self.query_one("#inp-epochs", Input).value
            batch = self.query_one("#inp-batch", Input).value
            layers = self.query_one("#inp-layers", Input).value
            hidden = self.query_one("#inp-hidden", Input).value
            lr = self.query_one("#inp-lr", Input).value
            augment = self.query_one("#inp-augment", Input).value
            kfolds = self.query_one("#inp-kfolds", Input).value
            
            cmd = f"python train_tsl51_v3.py --dataset {dataset} --model {model} --epochs {epochs} --batch {batch} --layers {layers} --hidden {hidden} --lr {lr} --augment {augment}"
            
            if kfolds.strip():
                cmd += f" --folds {kfolds}"
            
            return cmd
        except Exception:
            logger.exception("Failed to build training command")
            return "python train_tsl51_v3.py --epochs 10"
    
    @work(exclusive=True)
    async def _run_command_async(self, cmd: str, desc: str = "") -> None:
        """Run command asynchronously without blocking UI.

        This runs inside a Textual Worker. We keep a reference to the
        spawned subprocess so stop can terminate it.
        """
        self.training_active = True

        btn_run = self.query_one("#btn-run", Button)
        btn_stop = self.query_one("#btn-stop", Button)

        btn_run.disabled = True
        btn_stop.disabled = False
        self._update_status(f"[yellow]Running: {desc or cmd[:50]}...[/yellow]")

        self._log(f"\n[bold cyan]>>> {desc or 'Command'}[/bold cyan]")
        self._log(f"[dim]{cmd}[/dim]")
        self._log("[yellow]Starting...[/yellow]")

        try:
            # Prefer create_subprocess_exec for safety when the command is a simple argv list.
            import shlex

            shell_operators = ['&', '|', '>', '<', ';', '$', '`', '*', '&&', '||']
            use_shell = any(op in cmd for op in shell_operators) or 'cd ' in cmd or cmd.strip().startswith('cmd ') or cmd.strip().startswith('powershell')

            if not use_shell:
                # Safely split into argv and exec
                try:
                    argv = shlex.split(cmd)
                except Exception:
                    argv = None

                if argv:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=Path(__file__).parent.parent,
                    )
                else:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=Path(__file__).parent.parent,
                    )
            else:
                # Complex command: fall back to shell mode
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=Path(__file__).parent.parent,
                )

            # remember process so stop can terminate it
            self._current_process = process

            # Read output line by line
            while True:
                if process.stdout is None:
                    break
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode('utf-8', errors='replace').rstrip('\n')
                if text:
                    # Color code based on content
                    lower = text.lower()
                    if any(x in lower for x in ["error", "exception", "traceback"]):
                        self._log(f"[red]{text}[/red]")
                    elif any(x in lower for x in ["epoch", "loss", "accuracy", "f1"]):
                        self._log(f"[cyan]{text}[/cyan]")
                    elif any(x in lower for x in ["complete", "saved", "best"]):
                        self._log(f"[green]{text}[/green]")
                    else:
                        self._log(text)

            returncode = await process.wait()

            if returncode == 0:
                self._log("\n[bold green]Completed successfully![/bold green]")
                self._update_status("[green]Done![/green]")
            else:
                self._log(f"\n[bold red]Failed (exit code: {returncode})[/bold red]")
                self._update_status(f"[red]Failed (code {returncode})[/red]")

        except asyncio.CancelledError:
            # Worker was cancelled: ensure subprocess is killed
            try:
                if self._current_process:
                    self._current_process.kill()
                    await self._current_process.wait()
            except Exception:
                logger.exception("Error killing subprocess on cancel")
            self._log("\n[yellow]Cancelled by user[/yellow]")
            self._update_status("[yellow]Cancelled[/yellow]")
            raise
        except Exception as exc:
            logger.exception("Unexpected error while running command")
            self._log(f"\n[bold red]Error: {exc}[/bold red]")
            self._update_status("[red]Error[/red]")
        finally:
            # cleanup
            try:
                self._current_process = None
            except Exception:
                logger.debug("Failed clearing _current_process", exc_info=True)
            self.training_active = False
            try:
                btn_run.disabled = False
                btn_stop.disabled = True
            except Exception:
                logger.debug("Failed to reset run/stop buttons", exc_info=True)
            # clear stored worker reference if set
            try:
                self.worker = None
            except Exception:
                logger.debug("Failed clearing worker reference", exc_info=True)
    
    def action_run_training(self) -> None:
        """Run training with current config."""
        if self.training_active:
            self._update_status("[red]Already running![/red]")
            return
        
        cmd = self._get_training_command()
        desc = f"Training ({self.current_preset})"
        # start as a Textual worker and keep reference so we can cancel
        self.worker = self._run_command_async(cmd, desc)
    
    def action_stop_training(self) -> None:
        """Stop running process."""
        stopped = False
        # cancel the Textual worker if present
        if self.worker:
            try:
                self.worker.cancel()
                stopped = True
            except Exception as e:
                logger.exception("Failed to cancel worker: %s", e)

        # kill any running subprocess
        proc = getattr(self, "_current_process", None)
        if proc is not None:
            try:
                proc.kill()
                stopped = True
            except Exception as e:
                logger.exception("Failed to kill subprocess: %s", e)
            finally:
                try:
                    self._current_process = None
                except Exception:
                    logger.debug("Failed clearing _current_process during stop", exc_info=True)

        if stopped:
            self._update_status("[yellow]Stopping...[/yellow]")
    
    def action_clear_output(self) -> None:
        """Clear output log."""
        try:
            log = self.query_one("#output-log", Log)
            log.clear()
        except Exception:
            logger.exception("Failed to clear output log")
    
    async def action_quit(self) -> None:
        """Quit application."""
        if self.training_active:
            # attempt to stop running work
            try:
                self.action_stop_training()
            except Exception:
                logger.exception("Error while attempting to stop training on quit")
            # give control to event loop for cleanup
            await asyncio.sleep(0)
        self.exit()
    
    # Preset actions
    def action_preset_quick(self) -> None:
        self._load_preset("quick")
        self._log("[cyan]Loaded: Quick preset (5 epochs)[/cyan]")
    
    def action_preset_default(self) -> None:
        self._load_preset("default")
        self._log("[cyan]Loaded: Default preset (50 epochs)[/cyan]")
    
    def action_preset_full(self) -> None:
        self._load_preset("full_cv")
        self._log("[cyan]Loaded: Full K-Fold CV preset (5 folds)[/cyan]")
    
    def action_preset_mlp(self) -> None:
        self._load_preset("mlp")
        self._log("[cyan]Loaded: MLP Fast preset[/cyan]")
    
    def action_preset_large(self) -> None:
        self._load_preset("large_dataset")
        try:
            self.query_one("#sel-dataset", Select).value = "tsl51_expert_full"
        except Exception:
            pass
        self._log("[cyan]Loaded: Large Dataset preset (tsl51_expert_full)[/cyan]")
    
    def action_show_dashboard(self) -> None:
        """Show dashboard."""
        self._show_dashboard()
    
    def action_run_data_validation(self) -> None:
        """Run data validation."""
        self._run_data_validation()
    
    def action_run_model_benchmark(self) -> None:
        """Run model benchmark."""
        self._run_model_benchmark()
    
    # Button handlers
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # core actions
        if btn_id == "btn-run":
            self.action_run_training()
            return
        if btn_id == "btn-stop":
            self.action_stop_training()
            return
        if btn_id == "btn-clear":
            self.action_clear_output()
            return

        # presets
        if btn_id == "btn-preset-quick":
            self.action_preset_quick()
            return
        if btn_id == "btn-preset-default":
            self.action_preset_default()
            return
        if btn_id == "btn-preset-full":
            self.action_preset_full()
            return
        if btn_id == "btn-preset-mlp":
            self.action_preset_mlp()
            return
        if btn_id == "btn-preset-large":
            self.action_preset_large()
            return

        # navigation
        if btn_id == "btn-dashboard":
            self._show_dashboard()
            return
        if btn_id == "btn-data-validation":
            self._run_data_validation()
            return
        if btn_id == "btn-model-benchmark":
            self._run_model_benchmark()
            return

        # Generic command buttons -> id format: btn-<command_key>
        if btn_id.startswith("btn-"):
            cmd_key = btn_id[4:]
            meta = COMMANDS.get(cmd_key)
            if not meta:
                self._log(f"[red]Unknown command: {cmd_key}[/red]")
                return

            # If it's a training command, reuse training config
            if cmd_key.startswith("train_"):
                # set current command and run training builder
                self.current_cmd_id = cmd_key
                # if the command is train_mlp, switch model select
                if cmd_key == "train_mlp":
                    try:
                        self.query_one("#sel-model", Select).value = "mlp"
                    except Exception:
                        pass
                cmd = self._get_training_command()
                desc = meta.get("name", cmd_key)
                self.worker = self._run_command_async(cmd, desc)
                return

            # Non-training: run the defined command string
            cmd = meta.get("cmd")
            if not cmd:
                self._log(f"[red]No command configured for '{cmd_key}'[/red]")
                return
            desc = meta.get("name", cmd_key)
            self.worker = self._run_command_async(str(cmd), desc)
            return
    
    def _show_dashboard(self) -> None:
        """Show dashboard with system info."""
        self._log("\n[bold cyan]=== DASHBOARD ===[/bold cyan]")
        self._log("[cyan]System Information:[/cyan]")
        
        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                self._log(f"  GPU: {torch.cuda.get_device_name(0)}")
                self._log(f"  CUDA Available: Yes")
            else:
                self._log("  GPU: Not available (CPU only)")
        except Exception:
            self._log("  GPU: Unable to detect")
        
        # Dataset cache info
        try:
            from pathlib import Path
            cache_dir = Path(".cache/tsl51")
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.npz"))
                self._log(f"  Cached Datasets: {len(cache_files)} files")
                for f in cache_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    self._log(f"    - {f.name} ({size_mb:.1f} MB)")
            else:
                self._log("  Cached Datasets: None")
        except Exception:
            self._log("  Cached Datasets: Unable to check")
        
        # Model files
        try:
            from pathlib import Path
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pt"))
                self._log(f"  Trained Models: {len(model_files)} files")
                for f in sorted(model_files)[-5:]:  # Show last 5
                    self._log(f"    - {f.name}")
            else:
                self._log("  Trained Models: None")
        except Exception:
            self._log("  Trained Models: Unable to check")
        
        self._log("[cyan]Current Configuration:[/cyan]")
        try:
            dataset = self.query_one("#sel-dataset", Select).value
            model = self.query_one("#sel-model", Select).value
            epochs = self.query_one("#inp-epochs", Input).value
            batch = self.query_one("#inp-batch", Input).value
            self._log(f"  Dataset: {dataset}")
            self._log(f"  Model: {model}")
            self._log(f"  Epochs: {epochs}")
            self._log(f"  Batch Size: {batch}")
        except Exception:
            self._log("  Unable to read current config")
        
        self._log("[bold cyan]================[/bold cyan]")
    
    def _run_data_validation(self) -> None:
        """Run data validation on selected dataset."""
        if self.training_active:
            self._update_status("[red]Already running![/red]")
            return
        
        try:
            dataset = self.query_one("#sel-dataset", Select).value
            self._log(f"\n[bold cyan]=== DATA VALIDATION: {dataset} ===[/bold cyan]")
            
            cmd = f"python -c \"from src.data.loader import {self._get_loader_function(dataset)}; X, y, classes = {self._get_loader_function(dataset)}(); from src.data.loader import validate_dataset, print_dataset_quality_report; results = validate_dataset(X, y, classes); print_dataset_quality_report(results)\""
            
            desc = f"Data Validation ({dataset})"
            self.worker = self._run_command_async(cmd, desc)
        except Exception as e:
            self._log(f"[red]Error: {e}[/red]")
    
    def _run_model_benchmark(self) -> None:
        """Run model benchmark."""
        if self.training_active:
            self._update_status("[red]Already running![/red]")
            return
        
        try:
            dataset = self.query_one("#sel-dataset", Select).value
            self._log(f"\n[bold cyan]=== MODEL BENCHMARK: {dataset} ===[/bold cyan]")
            
            cmd = f"python benchmark_models.py --dataset {dataset}"
            
            desc = f"Model Benchmark ({dataset})"
            self.worker = self._run_command_async(cmd, desc)
        except Exception as e:
            self._log(f"[red]Error: {e}[/red]")
    
    def _get_loader_function(self, dataset: str) -> str:
        """Get the appropriate loader function name for a dataset."""
        loader_map = {
            "tsl51_user_sign": "load_tsl51_user_sign",
            "tsl51_expert": "load_tsl51_expert",
            "tsl51_expert_full": "load_tsl51_expert_full",
            "tsl51_combined": "load_tsl51_combined",
        }
        return loader_map.get(dataset, "load_tsl51_user_sign")


if __name__ == "__main__":
    app = TSLApp()
    app.run()
