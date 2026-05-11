from equity_project.src.get_data import get_data
from equity_project.src.run_backtest import run_backtest
from equity_project.src.train import train
from equity_project.src2.get_data_1_raw import load_data


def main():
    load_data()
    get_data()
    train()
    run_backtest()


if __name__ == "__main__":
    main()
