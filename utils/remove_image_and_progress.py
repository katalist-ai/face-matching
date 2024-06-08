import argparse
import json
import os

from . import prepare_number


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("numbers", nargs="+", type=int)
    return parser.parse_args().numbers


def remove_images(numbers):
    for number in numbers:
        os.remove(os.path.join("data/images", f"{number}.png"))


def remove_entries_from_json(numbers: list[str]):
    data = json.load(open("data/progress.json", "r"))
    for number in numbers:
        if number in data:
            del data[number]
    json.dump(data, open("data/progress.json", "w"), indent=2)


def main():
    numbers: list[str] = [prepare_number(n) for n in parse_arguments()]
    remove_images(numbers)
    remove_entries_from_json(numbers)


if __name__ == "__main__":
    main()
