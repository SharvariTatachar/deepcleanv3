import argparse
from pathlib import Path
from urllib.parse import quote

import requests


def read_filenames(ini_path):
    with open(ini_path, "r") as f:
        channels = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # Always skip the first channel (target/strain)
    channels = channels[1:]

    # Convert channel names to NPZ filenames
    filenames = [
        channel.replace(":", "_").replace("-", "_") + ".npz"
        for channel in channels
    ]

    return filenames


def download_file(share_url, filename, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # Don't re-download files that already exist
    if output_path.exists():
        print(f"Skipping existing: {filename}")
        return

    file_url = (
        share_url.rstrip("/")
        + "/download?path=%2F&files="
        + quote(filename)
    )

    print(f"Downloading: {filename}")

    with requests.get(file_url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help="Public CERNBox share URL",
    )

    parser.add_argument(
        "--ini",
        required=True,
        help="Channel list file",
    )

    parser.add_argument(
        "--output",
        default="./cernbox_data",
        help="Directory where files will be saved",
    )

    args = parser.parse_args()

    filenames = read_filenames(args.ini)

    print(f"Found {len(filenames)} witness channels")
    print("First few files:")
    for filename in filenames[:5]:
        print(f"  {filename}")

    for filename in filenames:
        try:
            download_file(
                args.url,
                filename,
                args.output,
            )

        except requests.RequestException as e:
            print(f"FAILED: {filename}")
            print(f"  {e}")


if __name__ == "__main__":
    main()