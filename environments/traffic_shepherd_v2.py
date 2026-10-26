"""Five semantic roles, local V2X/HDV observations and system rewards over TraCI."""
from collections import defaultdict, deque
from dataclasses import dataclass
import uuid

import numpy as np
import sumolib
import traci

from algorithms.traffic_shepherd_v2 import ACTION_NAMES, GLOBAL_DIM, OBS_DIM


@dataclass(frozen=True)
class RewardConfig:
    efficiency: float = 1.0
    safety: float = 10.0
    stability: float = 5.0
    fairness: float = 0.5
    control: float = 0.1


def reward_components(before, after, arrivals, collisions, commands, dt=0.1):
    """Nonnegative costs, with a signed efficiency benefit; no future data."""
    vehicles = list(after.values())
    humans = [s for n, s in after.items() if not n.startswith('cav')]
    count = max(1, len(vehicles))
    def mean(values):
        return float(np.mean(values)) if len(values) else 0.0
    speeds = [s['speed'] for s in vehicles]
    acceleration = [s['acceleration'] for s in vehicles]
    jerk = [(s['acceleration']-before[n]['acceleration'])/dt
            for n, s in after.items() if n in before]
    loss_increment = [max(0.0, s['loss']-before[n]['loss'])/dt
                      for n, s in after.items() if n in before]
    delay = [s['loss'] for s in humans]
    ttc_risk = [max(0.0, (2.5-s['ttc'])/2.5) for s in vehicles]
    return {
        'efficiency': mean(speeds)/20.0+len(arrivals)/count-mean(loss_increment),
        'safety': len(collisions)+mean(ttc_risk)+mean([float(a < -3.0) for a in acceleration]),
        'stability': (float(np.var(acceleration))/25.0 if acceleration else 0.0)+mean(np.abs(jerk))/50.0
                     +mean([s['speed_variance'] for s in vehicles])/100.0,
        'fairness': (mean(delay)+(float(np.percentile(delay, 90)) if delay else 0.0))/100.0,
        'control': mean([1.0+abs(a)/3.0 for a in commands.values() if a is not None])
                   *sum(a is not None for a in commands.values())/max(1, len(commands)),
    }


def weighted_reward(parts, weights):
    return (weights.efficiency*parts['efficiency']-weights.safety*parts['safety']
            -weights.stability*parts['stability']-weights.fairness*parts['fairness']
            -weights.control*parts['control'])


class TrafficShepherdV2Env:
    """Owns exactly one labeled SUMO connection. Five-CAV budget by default.

    No HDV actuation or forced lane changes. All observations are cached once per
    simulator step; repeatedly reading them cannot change temporal uncertainty.
    """
    def __init__(self, network, routes, seed=42, gui=False, max_controlled=5,
                 reward_config=None, action_mode='structured', scenario='ramp'):
        if max_controlled < 0 or action_mode not in ('structured', 'primitive'):
            raise ValueError('Invalid control budget or action mode')
        self.max_controlled, self.reward_config = max_controlled, reward_config or RewardConfig()
        self.action_mode, self.scenario = action_mode, scenario
        self.dt, self.steps = 0.1, 0
        self.previous_policies, self.controlled = {}, set()
        self.history = defaultdict(lambda: deque(maxlen=5))
        self.collisions, self.arrivals, self.last_commands = set(), [], {}
        self.last_parts = dict.fromkeys(('efficiency', 'safety', 'stability', 'fairness', 'control'), 0.0)
        self.road = sumolib.net.readNet(str(network))
        bottleneck = 'E1' if scenario == 'ramp' else 'merge_zone'
        self.bottleneck_x = self.road.getEdge(bottleneck).getFromNode().getCoord()[0]
        self.label = 'shepherd-v2-'+uuid.uuid4().hex
        traci.start(['sumo-gui' if gui else 'sumo', '-n', str(network), '-r', str(routes),
                     '--seed', str(seed), '--step-length', str(self.dt), '--no-step-log', 'true',
                     '--collision.action', 'remove', '--collision.check-junctions', 'true',
                     '--collision.mingap-factor', '0.0'], label=self.label)
        self.connection = traci.getConnection(self.label)
        self.states = self._snapshot()

    def _snapshot(self):
        v, states = self.connection.vehicle, {}
        living = set(v.getIDList())
        for name in sorted(living):
            speed, acceleration = v.getSpeed(name), v.getAcceleration(name)
            self.history[name].append((speed, acceleration))
            leader, follower = v.getLeader(name, 150.0), v.getFollower(name, 150.0)
            lead = leader[0] if leader and leader[0] in living else None
            follow = follower[0] if follower[0] in living else None
            gap = max(0.0, leader[1]) if lead else 150.0
            closing = speed-v.getSpeed(lead) if lead else 0.0
            lane = v.getLaneID(name)
            states[name] = {'speed': speed, 'acceleration': acceleration,
                'position': np.asarray(v.getPosition(name)), 'lane_index': v.getLaneIndex(name),
                'remaining': self.connection.lane.getLength(lane)-v.getLanePosition(name),
                'road': v.getRoadID(name), 'route': v.getRouteID(name),
                'max_speed': v.getAllowedSpeed(name), 'loss': v.getTimeLoss(name),
                'leader': lead, 'gap': gap, 'follower': follow,
                'back_gap': max(0.0, follower[1]) if follow else 150.0,
                'ttc': gap/closing if closing > 0 else float('inf'),
                'speed_variance': float(np.var([x[0] for x in self.history[name]])),
                'uncertainty': float(np.std([x[1] for x in self.history[name]]))}
        for name in list(self.history):
            if name not in living:
                del self.history[name]
        return states

    def controlled_ids(self):
        names = [n for n in self.states if n.startswith('cav')]
        return names[:self.max_controlled] if self.max_controlled else names

    def local_ids(self, name):
        position = self.states[name]['position']
        return [n for n, s in self.states.items()
                if np.linalg.norm(position-s['position']) <= 100.0]

    def observe(self):
        observations = {}
        for name in self.controlled_ids():
            s = self.states[name]
            x = s['position'][0]
            values = [s['speed']/30, s['acceleration']/5, s['remaining']/1000,
                      s['lane_index']/3, np.clip((self.bottleneck_x-x)/300, -1, 1),
                      float(s['route'] == 'route_ramp')]
            for other, gap in [(s['leader'], s['gap']), (s['follower'], s['back_gap'])]:
                speed = self.states[other]['speed'] if other else s['speed']
                values += [speed/30, min(gap, 150)/150, (speed-s['speed'])/30, float(other is not None)]
            nearby = self.local_ids(name)
            humans = [self.states[n] for n in nearby if not n.startswith('cav')]
            values += [float(np.mean([h['uncertainty'] for h in humans]))/5 if humans else 0.0,
                       float(np.std([h['speed'] for h in humans]))/10 if humans else 0.0,
                       float(np.mean([h['speed'] < 2 for h in humans])) if humans else 0.0, len(humans)/20]
            # Bounded V2X: three 100 m longitudinal zones, within +/-150 m.
            # This is local communication, not the centralized training state.
            for lower, upper in [(-150, -50), (-50, 50), (50, 150)]:
                zone = [t for t in self.states.values() if lower <= t['position'][0]-x < upper
                        and abs(t['position'][1]-s['position'][1]) <= 30]
                values += [len(zone)/20, float(np.mean([t['speed'] for t in zone]))/30 if zone else 0,
                           float(np.mean([t['speed'] < 2 for t in zone])) if zone else 0]
            neighbors = sorted((n for n in nearby if n != name and n.startswith('cav')),
                               key=lambda n: (np.linalg.norm(self.states[n]['position']-s['position']), n))[:3]
            for index in range(3):
                values.extend(self.previous_policies.get(neighbors[index], np.full(5, 0.2))
                              if index < len(neighbors) else np.zeros(5))
            observations[name] = np.clip(np.asarray(values, np.float32), -10, 10)
            if observations[name].shape != (OBS_DIM,):
                raise RuntimeError('Shepherd 2.0 observation schema mismatch')
        return observations

    def global_state(self):
        states = list(self.states.values())
        if not states:
            return np.zeros(GLOBAL_DIM, np.float32)
        speeds = [s['speed'] for s in states]
        accelerations = [s['acceleration'] for s in states]
        humans = [s['loss'] for n, s in self.states.items() if not n.startswith('cav')]
        return np.clip(np.asarray([len(states)/100, sum(n.startswith('cav') for n in self.states)/len(states),
            np.mean(speeds)/30, np.std(speeds)/10, np.mean(accelerations)/5, np.std(accelerations)/5,
            np.mean(humans)/100 if humans else 0, np.percentile(humans, 90)/100 if humans else 0,
            np.mean([s < 2 for s in speeds]), len(self.arrivals)/5, len(self.collisions)/5,
            np.mean([s['ttc'] < 2.5 for s in states])], np.float32), -10, 10)

    def command(self, name, action):
        if action not in range(len(ACTION_NAMES)):
            raise ValueError(f'Invalid role {action}')
        s = self.states[name]
        if self.action_mode == 'primitive':
            return (-1.5, -0.75, 0.0, 0.75, 1.5)[action]
        if action == 0:
            return None
        humans = [self.states[n] for n in self.local_ids(name) if not n.startswith('cav')]
        uncertainty = min(1.0, float(np.mean([h['uncertainty'] for h in humans]))/3 if humans else 0)
        leader_speed = self.states[s['leader']]['speed'] if s['leader'] else s['max_speed']
        if action == 1:  # Open a gap ahead of this CAV, with a gentle lower target.
            target = max(0.0, min(s['speed'], leader_speed)-2.0)
        elif action == 2:  # Damp a wave using the local HDV flow speed.
            target = float(np.mean([h['speed'] for h in humans])) if humans else leader_speed
        elif action == 3:  # Larger desired headway as human behavior varies.
            desired_gap = 2.5+(1.8+uncertainty)*s['speed']
            target = min(s['max_speed'], leader_speed+0.2*(s['gap']-desired_gap))
        else:  # Release accumulated delay smoothly, subject to the safety cap below.
            target = s['max_speed']
        acceleration = float(np.clip(0.5*(target-s['speed']), -3.0, 1.5))
        if action in (1, 3):
            acceleration *= 1.0-0.5*uncertainty
        # Braking-distance speed bound plus native SUMO safe speed enforcement. This
        # conservative heuristic is not a formal collision-avoidance guarantee.
        if s['leader']:
            safe_speed = np.sqrt(leader_speed**2+2*3.0*max(0.0, s['gap']-2.5))
            safe_accel = (safe_speed-s['speed'])/self.dt
            acceleration = min(acceleration, max(-3.0, safe_accel))
        return float(np.clip(acceleration, -3.0, 1.5))

    def step(self, actions, policies=None):
        if set(actions) != set(self.controlled_ids()):
            raise ValueError('Supply one action for every currently controlled CAV')
        v = self.connection.vehicle
        for name in (self.controlled-set(actions)) & set(self.states):
            v.setSpeedMode(name, 31)
            v.setSpeed(name, -1)
        commands = {name: self.command(name, action) for name, action in actions.items()}
        for name, acceleration in commands.items():
            v.setSpeedMode(name, 31)
            # setSpeed(-1) clears an earlier requested speed/acceleration before
            # returning to native IDM or issuing this step's new acceleration.
            v.setSpeed(name, -1)
            if acceleration is not None:
                v.setAcceleration(name, acceleration, duration=self.dt)
        self.controlled, self.last_commands = set(actions), commands
        self.previous_policies = {n: np.asarray(p, np.float32).copy() for n, p in (policies or {}).items() if n in actions}
        before = self.states
        self.connection.simulationStep()
        self.steps += 1
        self.collisions = set(self.connection.simulation.getCollidingVehiclesIDList())
        self.arrivals = list(self.connection.simulation.getArrivedIDList())
        self.states = self._snapshot()
        self.last_parts = reward_components(before, self.states, self.arrivals, self.collisions, commands, self.dt)
        reward = weighted_reward(self.last_parts, self.reward_config)
        # Team reward makes macroscopic HDV outcomes part of every CAV's return.
        # Explicit own collision penalty survives SUMO removing the agent.
        rewards = {n: reward-(10.0 if n in self.collisions else 0.0) for n in actions}
        return self.observe(), rewards

    def close(self):
        self.connection.close()
