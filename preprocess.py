from pathlib import Path
import snapatac2 as snap
from snapatac2.genome import hg38
import subprocess
import pandas as pd

narrowpeak_cols = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak"
]

def fragments_to_bedpe(input_file, output_file):
    command = [
        "macs3", "filterdup",
        "-i", input_file,
        "-f", "BED",
        "--keep-dup", "all",
        "-o", output_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def call_peaks(input_file, sample_name):
    command = [
        "macs3", "callpeak",
        "-t", input_file,
        "-f", "BEDPE",
        "-g", "hs",
        "-n", f"{sample_name}_data/bed/macs3_out/{sample_name}",
        "-B",
        "-q", "0.01",
        "--nomodel",
        "--call-summits",
        "--keep-dup", "all"
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully called peaks for {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def load_narrowpeak(narrowpeak_file: str) -> pd.DataFrame:
    return pd.read_csv(
        narrowpeak_file, sep="\t", header=None, names=narrowpeak_cols
    )


def filter_by_qvalue(peaks: pd.DataFrame, min_lgq: float = 2.0) -> pd.DataFrame:
    return peaks[peaks["qValue"] > min_lgq].reset_index(drop=True)


def remove_blacklisted_peaks(peaks: pd.DataFrame, blacklist_bed: str) -> pd.DataFrame:
    blacklist = pd.read_csv(
        blacklist_bed, sep="\t", header=None,
        usecols=[0, 1, 2], names=["chrom", "start", "end"]
    )

    merged = peaks.reset_index().merge(blacklist, on="chrom", suffixes=("", "_bl"))
    blacklisted_mask = (
        (merged["start"] < merged["end_bl"]) &
        (merged["end"] > merged["start_bl"])
    )

    blacklisted_original_idx = merged.loc[blacklisted_mask, "index"].unique()
    return peaks.drop(index=blacklisted_original_idx).reset_index(drop=True)


def write_narrowpeak(peaks: pd.DataFrame, output_file: str):
    peaks.to_csv(output_file, sep="\t", header=False, index=False)


if __name__ == "__main__":
    instruction = ""
    while type(instruction) != int:
        instruction = input("Are you converting from fragments or narrowPeak files? (1/2)\n")
        try:
            instruction = int(instruction)
        except ValueError:
            print("Invalid input. Please enter an integer.\n")

    sampleName = input("Please enter desired sample name:\n")

    if instruction == 1:
        fragments_to_bedpe(f"{sampleName}_data/fragments/fragments.tsv", f"{sampleName}_data/bed/{sampleName}.bedpe")
        call_peaks(f"{sampleName}_data/bed/{sampleName}.bedpe", sampleName)
    else:
        if instruction != 2:
            print("Invalid input. Please enter 1 or 2.\n")
            print("Exiting...")
            quit()

    peaks = load_narrowpeak(f"{sampleName}_data/bed/macs3_out/{sampleName}_peaks.narrowPeak")
    peaks = filter_by_qvalue(peaks, min_lgq=2.0)
    peaks = remove_blacklisted_peaks(peaks, "hg38_blacklist.bed")
    write_narrowpeak(peaks, f"{sampleName}_data/bed/macs3_out/{sampleName}_filtered.narrowPeak")





