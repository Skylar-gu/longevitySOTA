## Project

**Preserving Attractor Stability Under Progressive Neural Replacement in Wilson–Cowan Graph Networks**

---

## Objective

Simulate progressive neuron replacement in a graph-structured Wilson–Cowan network and quantify when and how attractor dynamics degrade. Identify replacement strategies that preserve stability, coherence, and memory-like states.

---

## Model

### Network structure

* Nodes: brain regions or neuron populations
* Graph: weighted adjacency matrix (W)

  * Source: synthetic small-world / modular graph or connectome-derived (Allen / HCP)
* Each node has two state variables:

  * (E_i(t)): excitatory activity
  * (I_i(t)): inhibitory activity

---

### Dynamics (discrete-time Wilson–Cowan)

For each node (i):

E_i(t+1) = (1 - \Delta t) E_i(t) + \Delta t \cdot \sigma\Big( w_{EE} \sum_j W_{ij} E_j(t) - w_{EI} I_i(t) + P_i \Big)

I_i(t+1) = (1 - \Delta t) I_i(t) + \Delta t \cdot \sigma\Big( w_{IE} E_i(t) - w_{II} I_i(t) + Q_i \Big)

* (\sigma): sigmoid or tanh
* (P_i, Q_i): external input / bias
* Parameters tuned to produce:

  * multistability or oscillatory regimes

---

## Attractor Definition

Define attractors as:

* stable fixed points or limit cycles in ((E, I)) space
* corresponding to:

  * memory-like states (distinct activity patterns)
  * oscillatory regimes (frequency bands)

---

## Initialization of Attractors

* Construct multiple initial conditions (x^{(k)} = (E^{(k)}, I^{(k)}))
* Verify convergence to distinct attractors:

  * cluster final states via distance metric
* Store:

  * attractor states (A_k)
  * basin regions via perturbation sampling

---

## Replacement Mechanism

### Progressive replacement parameter

* ( \alpha \in [0, 1] ): fraction of nodes replaced

---

### Replacement unit (per node)

For node (i), replace original update with surrogate:

(E_i(t+1), I_i(t+1)) = g_i\big(E_{\mathcal{N}(i)}(t), I_{\mathcal{N}(i)}(t)\big)

* (g_i): learned function (small MLP or linear + nonlinearity)
* Input:

  * neighboring node states
  * optionally self-state

---

### Training data

* Generate trajectories from intact network:

  * multiple initial conditions near each attractor
  * include perturbations
* Dataset:

  * inputs: local neighborhood states at time (t)
  * targets: true (E_i(t+1), I_i(t+1))

---

### Replacement policies

1. **Random**

   * uniform random node selection

2. **Centrality-based**

   * rank nodes by:

     * degree
     * eigenvector centrality
   * replace high-importance nodes first or last

3. **Low-impact-first**

   * estimate node importance via ablation:

     * remove node temporarily
     * measure change in attractor stability
   * replace least impactful nodes first

---

## Evaluation Pipeline

For each:

* replacement policy
* replacement fraction ( \alpha )

Run:

1. Replace nodes
2. Simulate dynamics from perturbed initial conditions
3. Evaluate metrics

---

## Metrics

### 1. Attractor persistence

* Initialize near (A_k)
* Run dynamics
* Check convergence to same attractor

Output:

* recovery rate per attractor

---

### 2. Basin size

* Perturb initial state with noise of varying magnitude
* Estimate probability of returning to (A_k)

Output:

* basin size vs ( \alpha )

---

### 3. Attractor drift

* Measure distance between:

  * original attractor (A_k)
  * post-replacement attractor (A_k^\alpha)

---

### 4. Dynamical stability

* Perturb trajectory mid-simulation
* Measure:

  * divergence
  * recovery time

---

### 5. Oscillatory structure (optional)

* Compute power spectrum of (E(t))
* Track:

  * dominant frequency shifts
  * coherence across nodes

---

### 6. Entropy (optional)

* Estimate entropy of network state distribution
* Track changes with increasing ( \alpha )

---

### 7. Tipping point

Define:

* ( \alpha_c ): smallest ( \alpha ) where:

  * recovery rate drops below threshold (e.g. 80%)

---

## Outputs

### Core plots

1. Recovery rate vs replacement fraction
2. Basin size vs replacement fraction
3. Attractor drift vs replacement fraction
4. Tipping point comparison across policies

---

### Visualizations

* State trajectories projected via PCA/UMAP
* Attractor clusters before and after replacement
* Heatmap:

  * x-axis: noise level
  * y-axis: replacement fraction
  * color: recovery probability

---

## Experimental Conditions

* Vary:

  * graph topology (random, modular, connectome-based)
  * coupling strength parameters
  * surrogate model capacity

---

## Minimal Implementation Stack

* Python
* NumPy / PyTorch
* NetworkX (graph construction)
* optional: PyTorch Geometric

---

## Execution Plan

### Phase 1

* Implement Wilson–Cowan network
* Verify multistability or oscillations

### Phase 2

* Identify and store attractors
* Validate basin structure

### Phase 3

* Generate trajectory dataset
* Train surrogate models

### Phase 4

* Implement progressive replacement
* Run evaluation pipeline

### Phase 5

* Compare replacement policies
* Produce plots and tipping points

---

## Key Result Targets

* Identification of critical replacement fraction ( \alpha_c )
* Demonstration of non-linear collapse in attractor stability
* Evidence that replacement order affects stability
* Evidence that local surrogate accuracy does not guarantee global stability

---

## Framing Statement

This project models neural replacement as a perturbation to a nonlinear dynamical system and evaluates safety in terms of preservation of attractor structure. It provides a framework for identifying stability boundaries and safe intervention trajectories in graph-based neural systems.
