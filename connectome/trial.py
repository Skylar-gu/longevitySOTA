import numpy as np
import networkx as nx

class WCGraph:
    def __init__(self, G, params):
        self.A  = nx.to_numpy_array(G, dtype=float)
        self.p  = params
        n = len(G)
        self.E  = np.random.rand(n) * 0.1
        self.I  = np.random.rand(n) * 0.05

    def F(self, x):
        return np.maximum(0.0, np.tanh(x))

    def step(self, dt=0.01, P=None, Q=None):
        p = self.p
        E_in = self.A @ self.E
        Pe = P if P is not None else p.get('PE', 0.5)
        Qi = Q if Q is not None else p.get('PI', 0.3)

        dE = (-self.E + (1 - p['rE']*self.E) * self.F(p['cEE']*E_in - p['cIE']*self.I + Pe))
        dI = (-self.I + (1 - p['rI']*self.I) * self.F(p['cEI']*self.E - p['cII']*self.I + Qi))

        self.E = np.clip(self.E + dt * dE / p['tauE'], 0, 1)
        self.I = np.clip(self.I + dt * dI / p['tauI'], 0, 1)

params = dict(cEE=2.0, cIE=2.5, cEI=3.0, cII=0.5,
              rE=0.1, rI=0.1, tauE=1.0, tauI=0.5,
              PE=0.5, PI=0.3)

G = nx.barabasi_albert_graph(30, 2, seed=42)
net = WCGraph(G, params)

trajectory = []
for _ in range(2000):
    net.step(dt=0.01)
    trajectory.append(net.E.copy())
trajectory = np.array(trajectory)

