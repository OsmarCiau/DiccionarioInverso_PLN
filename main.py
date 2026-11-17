import pathlib
import sys

src_path = str(pathlib.Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from reverse_dictionary.main import main

if __name__ == "__main__":
    main()
