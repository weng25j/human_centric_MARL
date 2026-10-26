"""One SUMO/TraCI environment and identical reward for every comparison learner."""
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from common.benchmark import OBS_DIMS, shared_reward
from environments.traffic_shepherd_v2 import TrafficShepherdV2Env

MAX_AGENTS = 5
HISTORY = 5
PRIMITIVE = (-1.5, -0.75, 0.0, 0.75, 1.5)


@dataclass(frozen=True)
class Control:
    acceleration: float = None
    lane_delta: int = 0
    target_speed: float = None


class CommonRewardMergeEnv(TrafficShepherdV2Env):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, 'connection'):
                self.connection.close()
            raise

    def command(self, name, control):
        if not isinstance(control, Control):
            raise TypeError('Comparison actions must be physical Control objects')
        if control.lane_delta not in (-1, 0, 1):
            raise ValueError('Lane delta must be -1, 0 or 1')
        if control.target_speed is not None:
            speed = self.states[name]['speed']
            return float(np.clip((control.target_speed-speed)/self.dt, -3.0, 1.5))
        if control.acceleration is not None:
            if not np.isfinite(control.acceleration) or not -3.0 <= control.acceleration <= 1.5:
                raise ValueError('Acceleration must be finite and in [-3,1.5]')
            return float(control.acceleration)
        return 0.0 if control.lane_delta else None

    def lane_mask(self, name):
        state = self.states[name]
        lanes = self.connection.edge.getLaneNumber(state['road'])
        return np.asarray([True, state['lane_index']+1 < lanes, state['lane_index'] > 0], bool)

    def step(self, actions, policies=None):
        if set(actions) != set(self.controlled_ids()):
            raise ValueError('Supply exactly the currently controlled CAVs')
        # Validate ALL controls before actuating anything.
        for name, control in actions.items():
            self.command(name, control)
            if control.lane_delta and not self.lane_mask(name)[1 if control.lane_delta > 0 else 2]:
                raise ValueError(f'Illegal lane change for {name}')
        for name, control in actions.items():
            if control.lane_delta:
                v = self.connection.vehicle
                v.setLaneChangeMode(name, 1621)  # native collision and gap checks
                v.changeLane(name, self.states[name]['lane_index']+control.lane_delta, self.dt)
        observations, _ = super().step(actions, policies)
        self.team_reward = shared_reward(self.last_parts, self.collisions, self.last_commands)
        return observations, dict.fromkeys(actions, self.team_reward)


@dataclass
class Frame:
    ids: tuple
    observations: np.ndarray
    histories: np.ndarray
    present: np.ndarray
    global_state: np.ndarray
    positions: np.ndarray
    lane_masks: np.ndarray
    game_prior: np.ndarray
    states: dict


class Observer:
    """Stable vehicle slots and bounded recurrent histories; no slot-ID leakage."""
    def __init__(self, method):
        self.method, self.ids = method, [None]*MAX_AGENTS
        self.histories = defaultdict(lambda: deque(maxlen=HISTORY))
        self.trust, self.previous, self.last_step = {}, {}, -1
        self.previous_actions = {}

    def record_actions(self, frame, actions):
        self.previous_actions = {n: int(actions[i]) for i, n in enumerate(frame.ids) if n is not None}

    def _trust_update(self, env):
        for name in env.controlled_ids():
            for other in env.local_ids(name):
                if other == name:
                    continue
                pair = name, other
                score = self.trust.get(pair, 0.5)
                if other in self.previous:
                    old = self.previous[other]
                    current = env.states[other]
                    # Paper leaves perceived cooperation task-dependent: here it
                    # means no hard braking/critical TTC and no abrupt cut-in.
                    cooperative = (current['acceleration'] >= -3 and current['ttc'] >= 1.5
                                   and abs(current['lane_index']-old['lane_index']) <= 1)
                    score = 0.4*score+0.6*float(cooperative)  # paper Eq.17, phi=.6
                self.trust[pair] = score
        alive = set(env.states)
        self.trust = {pair: score for pair, score in self.trust.items() if set(pair) <= alive}

    def _z_observation(self, env, name):
        s = env.states[name]
        def time_to_merge(t):
            return (env.bottleneck_x-t['position'][0])/max(0.1, t['speed'])/30.0
        vector = [s['speed']/30, s['acceleration']/5, s['lane_index']/3, time_to_merge(s)]
        nearest = sorted((n for n in env.local_ids(name) if n != name),
                         key=lambda n: (np.linalg.norm(env.states[n]['position']-s['position']), n))[:6]
        for index in range(6):
            if index >= len(nearest):
                vector.extend([0.0]*5)
                continue
            other = env.states[nearest[index]]
            vector.extend([other['speed']/30, other['acceleration']/5, other['lane_index']/3,
                           time_to_merge(other), np.linalg.norm(other['position']-s['position'])/150])
        zones = [('E0', 'approach'), ('E1', 'merge_zone'), ('E_ramp',)]
        for edges in zones:
            zone = [t for t in env.states.values() if t['road'] in edges]
            vector.extend([float(np.mean([t['speed'] for t in zone]))/30 if zone else 0, len(zone)/20])
        for edges in (zones[0], zones[2]):
            vector.append(sum(t['speed'] < 2 for t in env.states.values() if t['road'] in edges)/20)
        return vector

    def _game_prior(self, env, name):
        """Trust-triggered two-player keep/change game, using observable kinematics.

        Unspecified paper payoff regression coefficients are replaced by explicit
        one-second kinematic utilities. This is a declared benchmark adaptation.
        The external learning reward is never changed by this module.
        """
        from algorithms.trust_game import trust_lane_prior
        s = env.states[name]
        neighbors = [n for n in env.local_ids(name) if n != name]
        candidates = sorted(neighbors, key=lambda n: np.linalg.norm(env.states[n]['position']-s['position']))
        partner = next((n for n in candidates if env.states[n]['position'][0] <= s['position'][0]), None)
        if partner is None:
            return np.zeros(3, np.float32)
        other = env.states[partner]
        return trust_lane_prior(s, other, self.trust.get((name, partner), .5), partner.startswith('cav'), env.lane_mask(name))

    def capture(self, env):
        if self.last_step == env.steps:
            raise RuntimeError('Capture once per simulator step; reuse the returned frame')
        self.last_step = env.steps
        self._trust_update(env)
        live = set(env.controlled_ids())
        self.ids = [n if n in live else None for n in self.ids]
        for name in sorted(live-set(self.ids)):
            self.ids[self.ids.index(None)] = name
        for name in list(self.histories):
            if name not in live:
                del self.histories[name]
        base = env.observe()
        dim = OBS_DIMS[self.method]
        obs, history = np.zeros((MAX_AGENTS, dim), np.float32), np.zeros((MAX_AGENTS, HISTORY, dim), np.float32)
        positions, masks, prior = np.zeros((MAX_AGENTS, 2), np.float32), np.zeros((MAX_AGENTS, 3), bool), np.zeros((MAX_AGENTS, 3), np.float32)
        masks[:, 0] = True
        for slot, name in enumerate(self.ids):
            if name is None:
                continue
            s = env.states[name]
            vector = base[name].copy()
            if self.method == 'flat_mappo_common':
                lead, follow = s['leader'], s['follower']
                vector = [s['speed']/30, (1000-s['remaining'])/1000,
                          env.states[lead]['speed']/30 if lead else s['speed']/30, min(s['gap'], 150)/150,
                          env.states[follow]['speed']/30 if follow else s['speed']/30, min(s['back_gap'], 150)/150]
            elif self.method == 'z_merge_common':
                vector = self._z_observation(env, name)
            elif self.method == 'trust_marl_common':
                neighbors = sorted((n for n in env.local_ids(name) if n != name),
                                   key=lambda n: (np.linalg.norm(env.states[n]['position']-s['position']), n))[:3]
                vector = np.concatenate([vector, [self.trust.get((name, n), .5) for n in neighbors]+[.5]*(3-len(neighbors))])
                prior[slot] = self._game_prior(env, name)
            elif self.method == 'ma2c_common':
                # MA2C Eq.10: discount neighboring state signals. The identical
                # global reward is retained, with no spatial reward reshaping.
                others = sorted((n for n in live if n != name and n in env.local_ids(name)),
                                key=lambda n: (np.linalg.norm(env.states[n]['position']-s['position']), n))[:3]
                neighbor_state = [0.75*float(x) for n in others for x in base[n][:6]]
                vector = np.concatenate([vector, neighbor_state+[0.0]*(18-len(neighbor_state))])
            elif self.method == 'qmix_common':
                previous = np.zeros(5)
                if name in self.previous_actions:
                    previous[self.previous_actions[name]] = 1
                vector = np.concatenate([vector, previous])
            obs[slot] = np.clip(vector, -10, 10)
            self.histories[name].append(obs[slot].copy())
            values = np.stack(self.histories[name])
            history[slot, -len(values):] = values
            positions[slot], masks[slot] = s['position'], env.lane_mask(name)
        self.previous = env.states
        return Frame(tuple(self.ids), obs, history, np.asarray([n is not None for n in self.ids]),
                     env.global_state(), positions, masks, prior, env.states)


def controls_for(method, env, frame, actions, parameters=None):
    controls = {}
    for slot, name in enumerate(frame.ids):
        if name is None:
            continue
        action, state = int(actions[slot]), env.states[name]
        if method == 'shepherd_v2_common':
            # Use the original 2.0 role implementation, without duplicating it.
            controls[name] = Control(TrafficShepherdV2Env.command(env, name, action))
        elif method == 'z_merge_common':
            p = parameters[slot]
            if action < 2:
                controls[name] = Control(lane_delta=1 if action == 0 else -1)
            elif action == 2:
                controls[name] = Control(float(-.75+2.25*p[2]))
            elif action == 3:
                desired = 12.5+7.5*p[3]
                leader_speed = env.states[state['leader']]['speed'] if state['leader'] else state['max_speed']
                controls[name] = Control(float(np.clip(.5*(leader_speed-state['speed'])+.2*(state['gap']-desired), -3, 1.5)))
            else:
                controls[name] = Control()
        elif method == 'trust_marl_common':
            controls[name] = Control(float(parameters[slot]), (0, 1, -1)[action])
        elif method == 'nash_double_q_common':
            controls[name] = Control((-1.5, 0.0, 1.5)[action])
        elif method in ('flat_mappo_common', 'ma2c_common', 'qmix_common', 'maac_common'):
            controls[name] = Control(PRIMITIVE[action])
        elif method == 'rule_cav_common':
            controls[name] = Control(target_speed=max(0.0, state['speed']-2) if state['gap'] < 25 else 15.0)
        elif method == 'selfish_cav_common':
            controls[name] = Control(target_speed=state['max_speed'])
        else:
            controls[name] = Control()
    return controls
