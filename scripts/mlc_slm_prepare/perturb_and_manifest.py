import torchaudio
import subprocess
import random
import json
import os

from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import List
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Speed perturb WAV files and generate manifest.")
    parser.add_argument("--wav_scp", type=str, required=True, help="Path to wav.scp")
    parser.add_argument("--text", type=str, required=True, help="Path to text file")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Path to root of data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output audio")
    parser.add_argument("--manifest_all", type=str, required=True, help="Path to save the output manifest (all speeds)")
    parser.add_argument("--manifest_random", type=str, required=True, help="Path to save the output manifest (1 random speed)")
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.9, 1.0, 1.1], help="Speeds to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    return parser.parse_args()

def read_text_file(path):
    transcripts = dict()
    bad = set()
    with open(path, 'r') as reader:
        for f in reader:
            test = f.strip()
            splits = test.split(" ", 1)
            if len(splits) != 2:
                bad.add(test)
                continue
            utt_id, text = splits
            transcripts[utt_id.strip()] = text.strip()
    return transcripts, bad

def prepare_output_dirs(base_dir: Path, speeds: List[float]):
    output_dirs = {}
    for speed in speeds:
        if speed != 1.0:
            speed_dir = base_dir / f"speed_{speed:.1f}"
            speed_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[speed] = speed_dir
    return output_dirs

def read_wav_scp(path):
    with open(path, 'r') as f:
        return [line.strip().split(maxsplit=1) for line in f if line.strip()]

def perturb_and_manifest(args):
    utt_id, wav_path, text, speed, output_dir = args

    try:
        if speed == 1.0:
            wav_info = torchaudio.info(wav_path)
            return utt_id, speed, {
                "id": utt_id,
                "audio_filepath": os.path.abspath(wav_path),
                "offset": 0.0,
                "duration": wav_info.num_frames / wav_info.sample_rate,
                "context": None,
                "text": text,
            }
        else:
            out_path = Path(output_dir) / f"{utt_id}.wav"
            subprocess.run(
                ["sox", wav_path, str(out_path), "speed", str(speed), "rate", "16k"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav_info = torchaudio.info(str(out_path))
            return utt_id, speed, {
                "id": f"{utt_id}_speed{speed:.1f}",
                "audio_filepath": os.path.abspath(str(out_path)),
                "offset": 0.0,
                "duration": wav_info.num_frames / wav_info.sample_rate,
                "context": None,
                "text": text,
            }
    except subprocess.CalledProcessError:
        print(f"Failed to perturb {utt_id} at speed {speed}")
        return None

def main():
    args = parse_args()
    text_dict, bad = read_text_file(args.text)
    if len(bad) > 0:
        print(f"{len(bad)} utterances; could not read text:")
        print(bad)

    output_dirs = prepare_output_dirs(Path(args.output_dir), args.speeds)
    wav_list = read_wav_scp(args.wav_scp)

    tasks = []
    for speed in args.speeds:
        for utt_id, wav_path in wav_list[:20]:
            if utt_id not in text_dict:
                continue
            wav_path = os.path.normpath(os.path.join(args.data_root_dir, wav_path.removeprefix("train_data/")))
            tasks.append((utt_id, wav_path, text_dict[utt_id], speed, output_dirs.get(speed, None)))

    results = []
    with Pool(args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(perturb_and_manifest, tasks), total=len(tasks)):
            if result:
                results.append(result)

    grouped = defaultdict(list)
    for utt_id, speed, entry in results:
        grouped[utt_id].append(entry)

    with open(args.manifest_all, 'w') as f_all, open(args.manifest_random, 'w') as f_rand:
        for utt_id, entries in grouped.items():
            # all manifest
            for entry in entries:
                json.dump(entry, f_all, ensure_ascii=False,)
                f_all.write("\n")
            # random choice manifest
            rand_entry = random.choice(entries)
            json.dump(rand_entry, f_rand, ensure_ascii=False,)
            f_rand.write("\n")

if __name__ == "__main__":
    main()
