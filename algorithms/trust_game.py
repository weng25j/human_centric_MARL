"""Trust-triggered cooperative and noncooperative two-player lane games."""
import numpy as np


def rectangular_equilibrium(payoffs):
    """Verified 2x4 mixed Nash game; no greedy fallback labelled an equilibrium."""
    import nashpy
    a, b = np.asarray(payoffs, float)
    pure = np.argwhere((a >= a.max(axis=0, keepdims=True)-1e-9)
                       & (b >= b.max(axis=1, keepdims=True)-1e-9))
    if len(pure):
        i, j = max(pure.tolist(), key=lambda p: a[tuple(p)]+b[tuple(p)])
        return np.eye(2)[i], np.eye(4)[j]
    for row, col in nashpy.Game(a, b).support_enumeration():
        if not np.isfinite(row).all() or not np.isfinite(col).all():
            continue
        joint = np.outer(row, col)
        if (min(row) >= -1e-9 and min(col) >= -1e-9
                and abs(sum(row)-1) < 1e-6 and abs(sum(col)-1) < 1e-6
                and max((a@col).max()-(joint*a).sum(), (row@b).max()-(joint*b).sum()) < 1e-6):
            return row, col
    raise RuntimeError('Could not verify Trust-MARL lane-game equilibrium')


def lane_game(ego, follower):
    """Explicit one-second proxy for the paper's unspecified payoff coefficients."""
    payoff = np.zeros((2, 2, 4), float)
    separation = max(0.0, ego['position'][0]-follower['position'][0]-5)
    for change in range(2):
        for response, acceleration in enumerate((1.5, 0., -1.5, 0.)):
            gap = separation+ego['speed']-follower['speed']-.5*acceleration
            conflict = float(change and response != 3 and gap < 2.5+1.2*follower['speed'])
            speed_gain = (ego['max_speed']-ego['speed'])/30 if change else 0
            payoff[0, change, response] = .3*speed_gain-.3*conflict+.2*change-.2*float(not change and ego['route'] == 'route_ramp')
            payoff[1, change, response] = .3*(follower['speed']+acceleration)/30-.3*abs(acceleration)/5-.2*conflict-.2*float(response == 3)
    return payoff


def trust_lane_prior(ego, follower, trust, connected, legal):
    game = lane_game(ego, follower)
    cooperation = float(np.clip(trust-.4, 0, 1))
    if trust >= .4 and connected:
        # Cooperative agreement maximizes joint utility (paper Eq.21).
        utilities = game[0]+cooperation*game[1]
        change_probability = float(np.unravel_index(np.argmax(utilities), utilities.shape)[0])
    else:
        # The observed driver's acceleration supplies an incomplete-information
        # response prior; no ground-truth driver-type label is used.
        prior = np.full(4, .1)
        response = 0 if follower['acceleration'] > .1 else 2 if follower['acceleration'] < -.1 else 1
        prior[response] += .6
        conditioned = game.copy()
        conditioned[1] *= prior[np.newaxis, :]
        strategy, _ = rectangular_equilibrium(conditioned)
        change_probability = float(strategy[1])
    probabilities = np.asarray([max(.05, 1-change_probability), .05, .05])
    delta = follower['lane_index']-ego['lane_index']
    if delta in (-1, 1):
        direction = 1 if delta == 1 else 2
        if legal[direction]:
            probabilities[direction] = max(.05, change_probability)
    probabilities *= np.asarray(legal, bool)
    probabilities /= probabilities.sum()
    return np.log(np.maximum(probabilities, 1e-8)).astype(np.float32)
