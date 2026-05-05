import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import json
from datetime import datetime


METRICS = ("Loss", "Accuracy")
SPLITS = ("train", "val")


def compute_accuracy_milestones(df: pd.DataFrame) -> dict[str, float | int | None]:
	metrics_out: dict[str, float | int | None] = {}
	for split, key_prefix in [("train", "T_Tr"), ("val", "T_Te")]:
		acc_df = df[(df["metric"] == "Accuracy") & (df["split"] == split)].sort_values("step")
		if not acc_df.empty:
			base_wall = float(acc_df.iloc[0]["wall_time"])
			hit = acc_df[acc_df["value"] > 99]
			if not hit.empty:
				row = hit.iloc[0]
				metrics_out[f"{key_prefix}_i"] = int(row["step"])
				metrics_out[f"{key_prefix}_t"] = float(row["wall_time"] - base_wall)
				continue
		metrics_out[f"{key_prefix}_i"] = None
		metrics_out[f"{key_prefix}_t"] = None

	return metrics_out


def format_milestone_label(title: str, iteration: int | None, elapsed_seconds: float | None) -> str:
	if iteration is None or elapsed_seconds is None:
		return f"{title}: N/A"
	return f"{title} = {iteration} ({elapsed_seconds:.2f}s)"


def build_accuracy_comparison(df: pd.DataFrame) -> pd.DataFrame:
	train_acc = (
		df[(df["metric"] == "Accuracy") & (df["split"] == "train")][["step", "value", "wall_time"]]
		.rename(
			columns={
				"step": "train_step",
				"value": "train_acc",
				"wall_time": "train_wall_time",
			}
		)
		.sort_values("train_wall_time")
	)
	test_acc = (
		df[(df["metric"] == "Accuracy") & (df["split"] == "val")][["step", "value", "wall_time"]]
		.rename(
			columns={
				"step": "test_step",
				"value": "test_acc",
				"wall_time": "test_wall_time",
			}
		)
		.sort_values("test_wall_time")
	)

	if train_acc.empty or test_acc.empty:
		return pd.DataFrame(
			columns=[
				"test_step",
				"test_acc",
				"test_wall_time",
				"train_step",
				"train_acc",
				"train_wall_time",
				"time_delta",
				"step_delta",
				"test_time_s",
				"train_time_s",
				"acc_ratio",
			]
		)

	comparison = pd.merge_asof(
		test_acc,
		train_acc,
		left_on="test_wall_time",
		right_on="train_wall_time",
		direction="nearest",
	)
	comparison = comparison[comparison["train_acc"] != 0].copy()
	comparison["time_delta"] = comparison["test_wall_time"] - comparison["train_wall_time"]
	comparison["step_delta"] = comparison["test_step"] - comparison["train_step"]
	comparison["test_time_s"] = comparison["test_wall_time"] - float(comparison["test_wall_time"].iloc[0])
	comparison["train_time_s"] = comparison["train_wall_time"] - float(comparison["test_wall_time"].iloc[0])
	comparison["acc_ratio"] = (comparison["test_acc"] + 1e-8) / (comparison["train_acc"] + 1e-8)
	return comparison


def load_comparison_df(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)


def plot_comparison_dfs(comparison_df_paths: list[Path], output_path: Path | None = None) -> dict[str, plt.Figure]:
	sns.set_theme(style="whitegrid", context="talk")

	if not comparison_df_paths:
		raise ValueError("No comparison.csv files were provided")

	frames = []
	for path in comparison_df_paths:
		frame = load_comparison_df(path).copy()
		frame["source"] = path.stem
		frames.append(frame)

	combined = pd.concat(frames, ignore_index=True)
	figures: dict[str, plt.Figure] = {}
	plot_palette = sns.color_palette("deep", max(len(comparison_df_paths), 1))

	for figure_key, x_col, title, xlabel in [
		("step", "test_step", "Comparison Accuracy Across Test Step", "Test Step"),
		("time", "test_time_s", "Comparison Accuracy Across Test Time", "Test Time (s)"),
	]:
		fig, ax = plt.subplots(figsize=(15, 10))
		sns.lineplot(
			data=combined.sort_values(x_col),
			x=x_col,
			y="acc_ratio",
			hue="source",
			palette=plot_palette,
			linewidth=6,
			marker=None,
			ax=ax,
		)
		ax.axhline(1.0, color="black", linestyle="--", linewidth=2.5, alpha=0.7)
		ax.set_xlabel(xlabel, fontsize=32)
		ax.set_ylabel("Acc_test / Acc_train", fontsize=32)
		ax.legend(fontsize=20)
		ax.grid(True, alpha=0.3)
		ax.tick_params(axis="both", which="major", labelsize=32)
		ax.set_title(title, fontsize=36, pad=20, weight="bold")
		fig.tight_layout()
		figures[figure_key] = fig

	if output_path is not None:
		output_path.mkdir(parents=True, exist_ok=True)
		figures["step"].savefig(output_path / "comparison_acc_step.svg", bbox_inches="tight")
		figures["time"].savefig(output_path / "comparison_acc_time.svg", bbox_inches="tight")

	return figures


def load_scalar_records(log_dir: Path) -> pd.DataFrame:
	records = []

	for split in SPLITS:
		split_dir = log_dir / split
		if not split_dir.exists():
			raise FileNotFoundError(f"Missing TensorBoard split directory: {split_dir}")

		accumulator = event_accumulator.EventAccumulator(
			str(split_dir),
			size_guidance={event_accumulator.SCALARS: 0},
		)
		accumulator.Reload()

		for metric in METRICS:
			if metric not in accumulator.Tags().get("scalars", []):
				continue

			for event in accumulator.Scalars(metric):
				value = event.value * 100 if metric == "Accuracy" else event.value
				records.append(
					{
						"step": event.step,
						"split": split,
						"metric": metric,
						"value": value,
						"wall_time": event.wall_time,
					}
				)

	if not records:
		raise ValueError(f"No scalar events found under {log_dir}")

	return pd.DataFrame.from_records(records)


def plot_metrics(df: pd.DataFrame, output_path: Path | None = None, suffix: str = "") -> dict[str, plt.Figure]:
	sns.set_theme(style="whitegrid", context="talk")

	figures = {}
	lw = 6
	plot_palette = sns.color_palette("deep", 2)

	for metric in METRICS:
		subset = df[df["metric"] == metric]
		milestones = compute_accuracy_milestones(df)

		fig, ax = plt.subplots(figsize=(15, 10))
		sns.lineplot(
			data=subset.sort_values("step"),
			x="step",
			y="value",
			hue="split",
			hue_order=["train", "val"],
			palette=plot_palette,
			linewidth=lw,
			marker=None,
			ax=ax,
		)

		ax.set_xlabel("Epoch", fontsize=32)
		ax.set_ylabel("Accuracy (%)" if metric == "Accuracy" else "Loss", fontsize=32)
		ax.legend(fontsize=32)
		if metric == "Accuracy":
			milestone_text = "\n".join(
				[
					format_milestone_label("T_train", milestones.get("T_Tr_i"), milestones.get("T_Tr_t")),
					format_milestone_label("T_test", milestones.get("T_Te_i"), milestones.get("T_Te_t")),
				]
			)
			ax.text(
				0.98,
				0.98,
				milestone_text,
				transform=ax.transAxes,
				va="top",
				ha="right",
				fontsize=22,
				bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.5"},
			)
		ax.grid(True, alpha=0.3)
		ax.tick_params(axis="both", which="major", labelsize=32)
		ax.set_title(f"Training Progress: {metric}", fontsize=36, pad=20, weight="bold")
		fig.tight_layout()
		figures[metric.lower()] = fig

	comparison_df = build_accuracy_comparison(df)
	comparison_fig, comparison_ax = plt.subplots(figsize=(15, 10))
	sns.lineplot(
		data=comparison_df,
		x="test_step",
		y="acc_ratio",
		color=plot_palette[0],
		linewidth=lw,
		marker=None,
		ax=comparison_ax,
	)
	comparison_ax.axhline(1.0, color="black", linestyle="--", linewidth=2.5, alpha=0.7)
	comparison_ax.set_xlabel("Epoch", fontsize=32)
	comparison_ax.set_ylabel("Acc_test / Acc_train", fontsize=32)
	comparison_ax.tick_params(axis="both", which="major", labelsize=32)
	comparison_ax.grid(True, alpha=0.3)
	comparison_ax.set_title("Accuracy Ratio Across Iterations", fontsize=36, pad=20, weight="bold")
	comparison_fig.tight_layout()
	figures["comparison_acc"] = comparison_fig

	if output_path is not None:
		output_path.mkdir(parents=True, exist_ok=True)
		print(f"Saving figures to {output_path} with suffix '{suffix}'")
		for metric_name, fig in figures.items():
			if metric_name == "comparison_acc":
				fig.savefig(output_path / "comparison_acc.svg", bbox_inches="tight")
			else:
				fig.savefig(output_path / f"{metric_name}_{suffix}.svg", bbox_inches="tight")

		# Save a snippet of TrainerConfig for quick reference (if available)
		cfg_path = Path("trainer_config.py")
		if cfg_path.exists():
			with cfg_path.open("r") as cf:
				config_code = cf.readlines()
				with open(output_path / f"trial", "w") as f:
					f.writelines(config_code[10:37])

		# Compute first-iteration and wall-time metrics for hitting >99% accuracy
		metrics_out = compute_accuracy_milestones(df)
		comparison_df = build_accuracy_comparison(df)
		comparison_df_fname = f"comparison_{suffix if suffix else 'summary'}.csv"
		comparison_df.to_csv(output_path / comparison_df_fname, index=False)

		# Write metrics JSON
		metrics_fname = f"metrics_{suffix if suffix else 'summary'}.json"
		with open(output_path / metrics_fname, "w") as jf:
			json.dump(metrics_out, jf, indent=4)

		df.to_csv(output_path / f"records_{suffix if suffix else 'summary'}.csv", index=False)


	return figures


def main():
	parser = argparse.ArgumentParser(description="Plot TensorBoard scalars with seaborn.")
	parser.add_argument(
		"log_dir",
		type=Path,
		help="Parent TensorBoard log directory containing train/ and val/ subdirectories.",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=None,
		help="Directory to save separate loss and accuracy figures.",
	)
	parser.add_argument(
		"--suffix",
		type=str,
		default="",
		help="Suffix to append to the output filenames.",
	)
	args = parser.parse_args()

	df = load_scalar_records(args.log_dir)
	plot_metrics(df, args.output_dir, args.suffix)
	plt.show()


if __name__ == "__main__":
	main()
