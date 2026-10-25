"""Episode metrics for the new Nash merging comparison, using existing thresholds."""
import numpy as np

# Labels/units match the existing CSV/Excel report parser.
REPORT_METRICS = [
    ('throughput', 'Equivalent Hourly Throughput', 'veh/h'), ('arrivals', 'Raw Arrivals (Episode)', 'vehicles'),
    ('speed', 'Mean Speed', 'm/s'), ('travel_time', 'Mean Travel Time', 's'),
    ('collision_rate', 'Collision Rate', '%'), ('near_collisions', 'Avg Near-Collisions (TTC<1.5s)', 'per episode'),
    ('ttc', 'Avg TTC Violations (TTC<2.5s)', 'per episode'), ('hard_braking', 'Avg Hard Braking', 'per episode'),
    ('acceleration_variance', 'Acceleration Var', 'm/s^2'), ('jerk', 'Mean Abs Jerk', 'm/s^3'),
    ('wave', 'Wave Intensity (Var)', '(speed variance)'), ('interventions', 'Raw Interventions (Sum)', 'steps'),
    ('per_cav', 'Interventions Per CAV', 'steps/CAV'), ('duty', 'Duty Cycle (% Active Control)', '%'),
    ('decel', 'Mean Decel Magnitude', 'm/s^2'), ('duration', 'Mean Duration', 's'),
    ('success', 'Success Rate', '%'), ('avg_queue', 'Avg Max Queue', 'vehicles'),
    ('max_queue', 'Absolute Max Queue', 'vehicles'), ('delay', 'Mean Delay', 's'),
    ('delay90', '90th-Percentile Delay', 's'),
]


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def mean_ci(values):
    return mean(values), float(1.96*np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else 0.0


class MergingMetrics:
    def __init__(self, connection, dt=0.1):
        self.c, self.dt = connection, dt
        self.track, self.seen, self.cavs, self.collided = {}, set(), set(), set()
        self.ramp, self.failed, self.merged = set(), set(), set()
        self.speeds, self.accelerations, self.jerks, self.travel, self.delays = [], [], [], [], []
        self.queues, self.speed_trace, self.decel_trace, self.jerk_trace = [], [], [], []
        self.braking, self.durations, self.magnitudes = {}, [], []
        self.active_ttc, self.active_near = set(), set()
        self.arrivals = self.ttc = self.near = self.hard = self.interventions = self.active_steps = 0
        self.steps = 0

    def step(self, collisions):
        self.steps += 1
        c, v = self.c, self.c.vehicle
        self.collided.update(collisions)
        self.seen.update(collisions)
        self.cavs.update(name for name in collisions if name.startswith('cav'))
        for name in collisions:
            if name in self.ramp:
                self.failed.add(name)
        arrivals = c.simulation.getArrivedIDList()
        self.arrivals += len(arrivals)
        self.merged.update(set(arrivals) & self.ramp)
        for name in arrivals:
            if name in self.track:
                item = self.track[name]
                self.travel.append((self.steps-item['entry'])*self.dt)
                if not name.startswith('cav'):
                    self.delays.append(item['loss'])
        step_speeds, step_decels, step_jerks = [], [], []
        vehicles = set(v.getIDList())
        for name in vehicles:
            speed, acceleration = v.getSpeed(name), v.getAcceleration(name)
            self.seen.add(name)
            item = self.track.setdefault(name, {'entry': self.steps, 'previous': acceleration,
                                               'loss': 0.0, 'speeds': []})
            jerk = (acceleration-item['previous'])/self.dt
            item['previous'], item['loss'] = acceleration, max(item['loss'], v.getTimeLoss(name))
            item['speeds'].append(speed)
            self.speeds.append(speed)
            self.accelerations.append(acceleration)
            self.jerks.append(jerk)
            step_speeds.append(speed)
            step_jerks.append(abs(jerk))
            if acceleration < 0:
                step_decels.append(-acceleration)
            if acceleration < -3.0:
                self.hard += 1
            if v.getRouteID(name) == 'route_ramp':
                self.ramp.add(name)
                if speed < 0.1:
                    self.failed.add(name)
            if name.startswith('cav'):
                self.cavs.add(name)
                self.active_steps += 1
                if acceleration < -0.1:
                    self.interventions += 1
                    self.magnitudes.append(-acceleration)
                    self.braking[name] = self.braking.get(name, 0)+1
                elif name in self.braking:
                    self.durations.append(self.braking.pop(name)*self.dt)
            leader = v.getLeader(name, 100.0)
            closing = speed-v.getSpeed(leader[0]) if leader else 0.0
            ttc = leader[1]/closing if leader and closing > 0 else float('inf')
            for threshold, active, counter in [(2.5, self.active_ttc, 'ttc'), (1.5, self.active_near, 'near')]:
                if ttc < threshold:
                    if name not in active:
                        setattr(self, counter, getattr(self, counter)+1)
                    active.add(name)
                else:
                    active.discard(name)
        for name in list(self.braking):
            if name not in vehicles:
                self.durations.append(self.braking.pop(name)*self.dt)
        maximum = 0
        for lane in c.lane.getIDList():
            queue = 0
            for name in c.lane.getLastStepVehicleIDs(lane):
                queue = queue+1 if v.getSpeed(name) < 2.0 else 0
                maximum = max(maximum, queue)
        self.queues.append(maximum)
        self.speed_trace.append(mean(step_speeds))
        self.decel_trace.append(mean(step_decels))
        self.jerk_trace.append(mean(step_jerks))

    def result(self):
        durations = self.durations+[n*self.dt for n in self.braking.values()]
        return {'throughput': self.arrivals*3600/max(self.steps*self.dt, self.dt), 'arrivals': self.arrivals,
                'speed': mean(self.speeds), 'travel_time': mean(self.travel),
                'collision_rate': 100*len(self.collided)/max(1, len(self.seen)),
                'near_collisions': self.near, 'ttc': self.ttc, 'hard_braking': self.hard,
                'acceleration_variance': float(np.var(self.accelerations)) if self.accelerations else 0.0,
                'jerk': mean(np.abs(self.jerks)),
                'wave': mean([np.var(t['speeds']) for t in self.track.values() if len(t['speeds']) > 10]),
                'interventions': self.interventions, 'per_cav': self.interventions/max(1, len(self.cavs)),
                'duty': 100*self.interventions/max(1, self.active_steps), 'decel': mean(self.magnitudes),
                'duration': mean(durations), 'success': 100*len(self.merged-self.failed)/len(self.ramp) if self.ramp else 100.0,
                'avg_queue': mean(self.queues), 'max_queue': max(self.queues, default=0),
                'delay': mean(self.delays), 'delay90': float(np.percentile(self.delays, 90)) if self.delays else 0.0}
