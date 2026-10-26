"""Train nash_double_q_common with the common SUMO reward contract."""
from training.train_common_reward import main


if __name__ == '__main__':
    main(method='nash_double_q_common')
