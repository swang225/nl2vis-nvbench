import pathlib

def root():
    return pathlib.Path(__file__).parent.resolve()