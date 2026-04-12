import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


METRICS = ("Loss", "Accuracy")
SPLITS = ("train", "val")


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
		ax.grid(True, alpha=0.3)
		ax.tick_params(axis="both", which="major", labelsize=32)
		ax.set_title(f"Training Progress: {metric}", fontsize=36, pad=20, weight="bold")
		fig.tight_layout()
		figures[metric.lower()] = fig

	if output_path is not None:
		output_path.mkdir(parents=True, exist_ok=True)
		print(f"Saving figures to {output_path} with suffix '{suffix}'")
		for metric_name, fig in figures.items():
			fig.savefig(output_path / f"{metric_name}_{suffix}.svg", bbox_inches="tight")
	
	f = open("trainer_config.py", "r")
	config_code = f.readlines()
	f.close()
	with open(output_path / f"trial", "w") as f:
		f.writelines(config_code[10:37])

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
