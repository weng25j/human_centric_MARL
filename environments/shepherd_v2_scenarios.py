"""Isolated SUMO inputs for Shepherd 2.0; original network/routes are read-only."""
from pathlib import Path
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch

from environments.nash_merge import directory

ROOT = Path(__file__).resolve().parents[1]
MIXTURES = {'default': (0.2, 0.6, 0.2), 'cautious': (0.6, 0.3, 0.1),
            'aggressive': (0.1, 0.3, 0.6)}


def prepare_scenario(workspace, demand, penetration, scenario='ramp', mixture='default'):
    if scenario not in ('ramp', 'lane_drop') or mixture not in MIXTURES:
        raise ValueError('Unknown scenario or HDV mixture')
    if demand <= 0 or not 0 <= penetration <= 1:
        raise ValueError('Demand must be positive and penetration in [0,1]')
    # The unchanged legacy trainer imports a learner with an import-time
    # torch.manual_seed(42). Route preparation must not reset a new policy's
    # sampling stream or overwrite the RNG restored from its checkpoint.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            from training.train_traffic_shepherd import generate_mixed_traffic_route
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    network_dir = workspace/'SUMO_network'
    if not network_dir.exists():
        shutil.copytree(ROOT/'SUMO_network', network_dir)
    with directory(workspace):
        generate_mixed_traffic_route(penetration, demand)
    original = network_dir/'highway_onramp_actual_shepherd.rou.xml'
    tree = ET.parse(original)
    if mixture != 'default':
        for kind, probability in zip(tree.getroot().find('vTypeDistribution'), MIXTURES[mixture]):
            kind.set('probability', str(probability))
    network = network_dir/'test.net.xml'
    if scenario == 'lane_drop':
        network = network_dir/'shepherd_v2_lane_drop.net.xml'
        if not network.exists():
            nodes = network_dir/'shepherd_v2_lane_drop.nod.xml'
            edges = network_dir/'shepherd_v2_lane_drop.edg.xml'
            connections = network_dir/'shepherd_v2_lane_drop.con.xml'
            nodes.write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="700" y="0"/>'
                             '<node id="c" x="1000" y="0"/><node id="d" x="1700" y="0"/></nodes>')
            edges.write_text('<edges><edge id="approach" from="a" to="b" numLanes="2" speed="30"/>'
                             '<edge id="merge_zone" from="b" to="c" numLanes="2" speed="30"/>'
                             '<edge id="exit" from="c" to="d" numLanes="1" speed="30"/></edges>')
            # The right lane drops at x=1000. Only the left upstream lane continues;
            # SUMO's native strategic lane changing performs the merge in 300 m.
            connections.write_text('<connections><connection from="approach" to="merge_zone" fromLane="0" toLane="0"/>'
                                   '<connection from="approach" to="merge_zone" fromLane="1" toLane="1"/>'
                                   '<connection from="merge_zone" to="exit" fromLane="1" toLane="0"/></connections>')
            subprocess.run(['netconvert', '--node-files', str(nodes), '--edge-files', str(edges),
                            '--connection-files', str(connections), '--output-file', str(network),
                            '--no-turnarounds', 'true'], check=True, capture_output=True, text=True)
        for route in tree.getroot().findall('route'):
            route.set('edges', 'approach merge_zone exit')
        for flow in list(tree.getroot().findall('flow')):
            if flow.get('route') == 'route_ramp':
                tree.getroot().remove(flow)
    routes = network_dir/'shepherd_v2.rou.xml'
    tree.write(routes, encoding='utf-8', xml_declaration=True)
    return network, routes
