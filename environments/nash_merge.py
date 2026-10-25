"""TraCI interaction games for the Nash Double-Q benchmark adaptation."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import traci

ACCELERATIONS = (-1.5, 0.0, 1.5)
ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def directory(path):
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def prepare_routes(workspace, demand, penetration):
    """Use Traffic Shepherd's exact traffic generator in an isolated workspace."""
    from training.train_traffic_shepherd import generate_mixed_traffic_route
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    network = workspace/'SUMO_network'
    if not network.exists():
        shutil.copytree(ROOT/'SUMO_network', network)
    with directory(workspace):
        generate_mixed_traffic_route(penetration, demand)
    return network/'test.net.xml', network/'highway_onramp_actual_shepherd.rou.xml'


def interaction_groups(snapshot, radius=100.0):
    """Disjoint nearest-CAV pairs; unmatched vehicles play a one-player game."""
    remaining = set(snapshot)
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        nearest = sorted(remaining, key=lambda other: (
            np.linalg.norm(snapshot[first]['position']-snapshot[other]['position']), other))
        second = None
        if nearest and np.linalg.norm(snapshot[first]['position']-snapshot[nearest[0]]['position']) <= radius:
            second = nearest[0]
            remaining.remove(second)
        groups.append((first, second))
    return groups


def encode_game(group, snapshot):
    values, presence = [], []
    masks = np.zeros((2, 3), bool)
    for role, vehicle in enumerate(group):
        if vehicle is None or vehicle not in snapshot:
            values.extend([0.0]*6)
            presence.append(0.0)
            masks[role, 1] = True
        else:
            state = snapshot[vehicle]
            values.extend(state['observation'])
            presence.append(1.0)
            masks[role] = [state['speed'] > 0.1, True, state['speed'] < state['max_speed']]
    return np.asarray(values+presence, np.float32), masks


class NashMergeEnv:
    def __init__(self, network, routes, seed=42, gui=False, step_length=0.1, max_controlled=5):
        self.max_controlled = max_controlled
        self.controlled = set()
        self.label = 'nash-'+uuid.uuid4().hex
        self.step_length = step_length
        traci.start(['sumo-gui' if gui else 'sumo', '-n', str(network), '-r', str(routes),
                     '--seed', str(seed), '--step-length', str(step_length), '--no-step-log', 'true',
                     '--collision.action', 'remove', '--collision.check-junctions', 'true',
                     '--collision.mingap-factor', '0.0'], label=self.label)
        self.connection = traci.getConnection(self.label)
        self.collisions = set()

    def snapshot(self):
        vehicle = self.connection.vehicle
        states = {}
        for name in vehicle.getIDList():
            if not name.startswith('cav'):
                continue
            if self.max_controlled and len(states) >= self.max_controlled:
                break
            speed = vehicle.getSpeed(name)
            leader, follower = vehicle.getLeader(name, 150.0), vehicle.getFollower(name, 150.0)
            ls, gap = (vehicle.getSpeed(leader[0]), leader[1]) if leader else (speed, 150.0)
            fs, back = (vehicle.getSpeed(follower[0]), follower[1]) if follower[0] else (speed, 150.0)
            states[name] = {'speed': speed, 'max_speed': vehicle.getAllowedSpeed(name),
                            'position': np.asarray(vehicle.getPosition(name)),
                            'observation': [speed/30.0, vehicle.getLanePosition(name)/1000.0,
                                            ls/30.0, min(gap, 150.0)/150.0,
                                            fs/30.0, min(back, 150.0)/150.0]}
        return states

    def reward(self, name):
        """Same individual human-centric reward terms/weights as Traffic Shepherd."""
        v = self.connection.vehicle
        speed, accel = v.getSpeed(name), v.getAcceleration(name)
        follower = v.getFollower(name, 100.0)[0]
        value = speed/20.0 - 0.1*max(0.0, -accel)/5.0
        if follower:
            braking = v.getAcceleration(follower)
            if braking < -4.5:
                value -= 10.0
            value -= max(0.0, -braking)
            value -= 0.5*v.getWaitingTime(follower)/100.0
        return float(value)

    def step(self, actions):
        v = self.connection.vehicle
        active = set(v.getIDList())
        for name in (self.controlled-set(actions)) & active:
            v.setSpeedMode(name, 31)
            v.setSpeed(name, -1)
        self.controlled = set(actions)
        for name, action in actions.items():
            # Match the other learned comparisons' explicit longitudinal control.
            v.setSpeedMode(name, 0)
            v.setAcceleration(name, ACCELERATIONS[action], duration=self.step_length)
        self.connection.simulationStep()
        self.collisions = set(self.connection.simulation.getCollidingVehiclesIDList())
        states = self.snapshot()
        living = set(v.getIDList())
        rewards = {name: -10.0 if name in self.collisions else self.reward(name) if name in living else 0.0
                   for name in actions}
        return states, rewards

    def close(self):
        self.connection.close()
