import os
import sys
import tempfile

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_analysis import analyse, load_spectrogram, results_to_bands
from timestamp_format import build_datafile, load_datafile, save_datafile
from visualiser_render import render_video

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_BANDS, F_LOW, F_HIGH = 20, 50.0, 7000.0
_EDGES = F_LOW * (F_HIGH / F_LOW) ** (np.arange(N_BANDS + 1) / N_BANDS)
DEFAULT_BANDS = [
    [f"b{i + 1:02d}", round(float(_EDGES[i]), 1), round(float(_EDGES[i + 1]), 1), 0.5]
    for i in range(N_BANDS)
]
HEADERS = ["name", "low_hz", "high_hz", "sensitivity"]


def _bands_from_table(table):
    df = pd.DataFrame(table, columns=HEADERS).dropna()
    bands = []
    for _, row in df.iterrows():
        low, high = float(row["low_hz"]), float(row["high_hz"])
        if high <= low:
            raise gr.Error("Each band needs high_hz greater than low_hz")
        bands.append({
            "name": str(row["name"]).strip() or f"band{len(bands)}",
            "low_hz": low,
            "high_hz": high,
            "sensitivity": float(np.clip(float(row["sensitivity"]), 0.0, 1.0)),
        })
    if not bands:
        raise gr.Error("Define at least one band")
    return bands


def _make_plot(results, duration, view_start, view_len):
    end = duration if view_len <= 0 else min(duration, view_start + view_len)
    start = min(view_start, max(0.0, end - 0.5))
    n = len(results)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, max(2.0, 1.4 * n)), squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        hop = duration / max(1, len(r["flux"]))
        t_axis = (np.arange(len(r["flux"])) + 0.5) * hop
        ax.plot(t_axis, r["flux"], color="tab:blue", lw=0.6)
        ax.plot(t_axis, r["threshold"], color="tab:orange", lw=0.8)
        sel = (r["times"] >= start) & (r["times"] <= end)
        ax.vlines(r["times"][sel], 0, 1.2, color="tab:red", lw=0.8)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel(r["name"], rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
        ax.text(0.995, 0.85, f"{len(r['times'])} events", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[-1, 0].set_xlim(start, end)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    return fig


def make_spectrogram(audio_path, table):
    if not audio_path:
        return None
    mag, freqs, hop_s, duration = load_spectrogram(audio_path)
    factor = max(1, int(np.ceil(mag.shape[1] / 3000)))
    cols = mag.shape[1] // factor * factor
    m = mag[1:, :cols].reshape(mag.shape[0] - 1, -1, factor).mean(axis=2)
    db = 20 * np.log10(m + 1e-6)
    t_edges = np.linspace(0, duration, db.shape[1] + 1)
    f = freqs[1:]
    f_edges = np.concatenate([[f[0] * 0.99], (f[:-1] + f[1:]) / 2, [f[-1] * 1.01]])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.pcolormesh(t_edges, f_edges, db, vmin=db.max() - 80, vmax=db.max(), cmap="magma", shading="flat", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(20, min(20000, f[-1]))
    ax.set_xlabel("seconds")
    ax.set_ylabel("Hz")
    try:
        for b in _bands_from_table(table):
            for edge in (b["low_hz"], b["high_hz"]):
                ax.axhline(edge, color="cyan", lw=0.8, ls="--")
            ax.text(duration * 0.995, np.sqrt(b["low_hz"] * b["high_hz"]), b["name"], color="white",
                    ha="right", va="center", fontsize=9)
    except (ValueError, TypeError, gr.Error):
        pass
    fig.tight_layout()
    return fig


def run_analysis(audio_path, table, global_offset, min_gap_ms, view_start, view_len):
    if not audio_path:
        raise gr.Error("Upload a wav file first")
    bands = _bands_from_table(table)
    results, duration, _ = analyse(audio_path, bands, global_offset, min_gap_ms / 1000.0)
    fig = _make_plot(results, duration, view_start, view_len)
    summary = pd.DataFrame({
        "band": [r["name"] for r in results],
        "range_hz": [f"{r['low_hz']:.0f}-{r['high_hz']:.0f}" for r in results],
        "events": [len(r["times"]) for r in results],
        "events_per_sec": [round(len(r["times"]) / duration, 2) for r in results],
    })
    settings = {
        "bands": bands,
        "global_offset": global_offset,
        "min_gap_ms": min_gap_ms,
    }
    data = build_datafile(os.path.basename(audio_path), duration, settings, results_to_bands(results))
    return fig, summary, data


def save_timestamps(data, audio_path):
    if not data:
        raise gr.Error("Run the analysis first")
    stem = os.path.splitext(os.path.basename(audio_path or "audio"))[0]
    path = os.path.join(OUT_DIR, f"{stem}_timestamps.json")
    save_datafile(data, path)
    return path


def load_timestamps(file):
    if file is None:
        raise gr.Error("Choose a timestamp datafile")
    try:
        data = load_datafile(file)
    except (ValueError, KeyError, OSError) as e:
        raise gr.Error(f"Could not load datafile: {e}")
    total = sum(len(b["events"]) for b in data["bands"])
    msg = f"Loaded {len(data['bands'])} bands, {total} events, duration {data['duration']:.1f} s, source {data['audio_file']}"
    return data, msg


def render(data, audio_path, width, height, fps, life_s, seed, progress=gr.Progress()):
    if not data:
        raise gr.Error("Run the analysis or load a datafile first")
    if not audio_path:
        raise gr.Error("Upload the wav file to use as the soundtrack")
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    out = tempfile.mktemp(prefix=f"{stem}_vis_", suffix=".mp4", dir=OUT_DIR)
    render_video(data, audio_path, out, int(width), int(height), int(fps), float(life_s), int(seed), progress)
    return out, out


with gr.Blocks(title="Audio Band Visualiser") as demo:
    gr.Markdown("# Audio Band Visualiser")
    data_state = gr.State(None)

    with gr.Row():
        audio_in = gr.Audio(label="Audio file (wav)", type="filepath", sources=["upload"])

    with gr.Tab("1. Analyse"):
        with gr.Row():
            with gr.Column(scale=1, min_width=420):
                table = gr.Dataframe(
                    value=DEFAULT_BANDS, headers=HEADERS, datatype=["str", "number", "number", "number"],
                    column_count=(4, "fixed"), row_count=(N_BANDS, "dynamic"), interactive=True,
                    label="Bands (sensitivity 0 to 1)",
                )
                global_offset = gr.Slider(-0.5, 0.5, value=0.0, step=0.01, label="Global sensitivity offset")
                min_gap = gr.Slider(20, 1000, value=100, step=10, label="Minimum gap between events in a band (ms)")
                with gr.Row():
                    view_start = gr.Number(value=0, label="Plot start (s)")
                    view_len = gr.Number(value=0, label="Plot window (s, 0 for entire length)")
                analyse_btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=3):
                spectrogram = gr.Plot(label="Spectrogram of entire file with band edges (dashed)")
        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(label="Band onset strength (blue), threshold (orange), events (red)")
            with gr.Column(scale=1, min_width=420):
                summary = gr.Dataframe(label="Events per band", interactive=False)
        with gr.Row():
            save_btn = gr.Button("Save timestamps")
            save_file = gr.File(label="Timestamp datafile")

    with gr.Tab("2. Visualise"):
        with gr.Row():
            load_file = gr.File(label="Load a saved timestamp datafile (optional, otherwise uses current analysis)", type="filepath")
            load_msg = gr.Textbox(label="Datafile status", interactive=False)
        with gr.Row():
            width = gr.Number(value=512, label="Size (square, px)", precision=0)
            height = width
            fps = gr.Number(value=30, label="FPS", precision=0)
            life = gr.Slider(0.1, 3.0, value=0.8, step=0.05, label="Object lifetime (s)")
            seed = gr.Number(value=1, label="Layout seed", precision=0)
        render_btn = gr.Button("Render mp4", variant="primary")
        video_out = gr.Video(label="Result", height=512, width=512)
        video_file = gr.File(label="Download mp4")

    plot_inputs = [audio_in, table, global_offset, min_gap, view_start, view_len]
    analyse_btn.click(run_analysis, plot_inputs, [plot, summary, data_state])
    audio_in.change(make_spectrogram, [audio_in, table], spectrogram)
    table.change(make_spectrogram, [audio_in, table], spectrogram)
    save_btn.click(save_timestamps, [data_state, audio_in], save_file)
    load_file.change(load_timestamps, load_file, [data_state, load_msg])
    render_btn.click(render, [data_state, audio_in, width, height, fps, life, seed], [video_out, video_file])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
