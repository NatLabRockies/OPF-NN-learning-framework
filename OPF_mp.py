###############################################################################
# DC3
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# DC3 also leverages a variety of third-party software packages, which have separate licensing policies.

# This class module ACOPFProblem was originally part of DC3, available: https://github.com/locuslab/DC3/tree/main
# Copied with modification from https://github.com/locuslab/DC3/blob/main/utils.py
###############################################################################

import sys
import os
import multiprocessing as mp
import time
import random
from contextlib import nullcontext

# Disable mpi4py auto-init for non-mpirun launches (quickstart pattern).
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("MPI4PY_RC_FINALIZE", "0")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from pathlib import Path

# Make local Optimization-Layers-for-Pyomo/src importable from this repo checkout.
_here = Path(__file__).resolve().parent
_candidates = [
    _here.parent / "Optimization-Layers-for-Pyomo" / "src",
    _here / "Optimization-Layers-for-Pyomo" / "src",
]
for _src in _candidates:
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
        break

# Compatibility shim: older PYPOWER expects numpy.Inf / numpy.NaN (removed in newer NumPy).
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from copy import deepcopy
from pypower.api import makeYbus, makeB
from pypower import idx_bus, idx_gen
import argparse
import pyomo.environ as pyo
from utility.PF_solver import PFFunction_eval 
    
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

from case_ACTIVSg2000 import case_ACTIVSg2000

from pyomolayers import PyomoOptLayer
try:
    from pyomolayers.parallel_layer import ParallelPyomoOptLayer
    HAS_MP = True
except Exception:
    ParallelPyomoOptLayer = None
    HAS_MP = False

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='ACOPF Solver')
parser.add_argument('--epochs', type=int, default=10, help='number of neural network epochs')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='neural network learning rate')
parser.add_argument('--hiddenSize', type=int, default=512, help='hidden layer size for neural network')
parser.add_argument('--num_workers', type=int, default=0, help='number of worker processes for ParallelPyomoOptLayer (0 = serial)')
parser.add_argument('--ipopt_print_level', type=int, default=0, help='IPOPT print level (0 keeps epoch logs visible)')

args = vars(parser.parse_args())
IS_POOL_WORKER = mp.current_process().name != "MainProcess"
if not IS_POOL_WORKER:
    print(args)

if __name__ != '__main__' or IS_POOL_WORKER:
    args['num_workers'] = 0
    args['epochs'] = 0

class ACOPFProblem:
    def __init__(self, data, ppc, num, train_frac=0.7, valid_frac = 0.1):
        self.nbus = ppc['bus'].shape[0]

        self.ppc = ppc
        self.baseMVA = ppc['baseMVA'] 
        self.device = None
        
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))
        self.allbus_idxes = np.arange(self.nbus)
        self.gen_bus_idx, self.gen_bus_unique, self.gen_bus_full, self.non_slack_gen_idx, self.gen_bus_nonslack, self.g_slack = self.gen_map_pv()
        
        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])
        self.pv_nonslack_idxes = np.array([np.where(x == self.nonslack_idxes)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.allgen_idxes = np.arange(self.ng)
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        self.nspv = len(self.spv)
        self.nbr = ppc['branch'].shape[0]
        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus
        
        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_costs = torch.tensor(ppc['gencost'][:,6], dtype=torch.get_default_dtype())

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.baseMVA, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.baseMVA, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.baseMVA, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.baseMVA, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        slackva_np = np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])
        self.slackva = torch.as_tensor(slackva_np, dtype=torch.get_default_dtype())
        self.bfmax = torch.tensor((ppc['branch'][:, 5] / self.baseMVA)**2, dtype=torch.get_default_dtype())
        # After PF analysis, generators at the same bus are aggregated to the bus level.
        # In the Pyomo model, the generator limits are enforced at the generator level level. 
        self.pg_min_spv, self.pg_max_spv, self.qg_min_spv, self.qg_max_spv = self.get_spv_bounds()
        
        ppc2 = deepcopy(ppc)
        ppc2['bus'][:, 0] -= 1
        ppc2['branch'][:, [0, 1]] -= 1
        Ybus, Yf, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        Yfbus = Yf.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())
        self.Yfbusr = torch.tensor(np.real(Yfbus), dtype=torch.get_default_dtype())
        self.Yfbusi = torch.tensor(np.imag(Yfbus), dtype=torch.get_default_dtype())

        Bp, Bpp = makeB(self.baseMVA, ppc2['bus'], ppc2['branch'], 2)
        Bp = Bp.todense()
        Bpp = Bpp.todense()
        Bp_reduced = Bp[np.concatenate([self.pv, self.pq]),:][:,np.concatenate([self.pv, self.pq])]
        Bpp_reduced = Bpp[np.concatenate([self.pq]),:][:,np.concatenate([self.pq])]
        self.Bp = torch.inverse(torch.tensor(Bp_reduced, dtype=torch.get_default_dtype()))
        self.Bpp = torch.inverse(torch.tensor(Bpp_reduced, dtype=torch.get_default_dtype()))

        self.F_BUS = ppc2['branch'][:, 0].astype(int)
        self.T_BUS = ppc2['branch'][:, 1].astype(int)
        self.Cf = torch.tensor(np.transpose(csr_matrix((np.ones(self.nbr), (range(self.nbr), self.F_BUS)), (self.nbr, self.nbus)).todense()), dtype=torch.get_default_dtype())
        self.Ct = torch.tensor(np.transpose(csr_matrix((np.ones(self.nbr), (range(self.nbr), self.T_BUS)), (self.nbr, self.nbus)).todense()), dtype=torch.get_default_dtype())
            
        ## Define optimization problem input and output variables
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.baseMVA
        voltage = data['Vol'].T
        # branch_flow = branch_flow / self.baseMVA / self.baseMVA

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)[:num,:]
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(voltage), np.angle(voltage)], axis=1)[:num,:]

        self.X = torch.tensor(X, dtype=torch.get_default_dtype())
        self.Y = torch.tensor(Y, dtype=torch.get_default_dtype())
        self.xdim = X.shape[1]
        self.ydim = Y.shape[1]
        self.device = self.X.device

        ## Define train split
        self.trainX = self.X[:int(self.X.shape[0] * train_frac)]
        self.trainY = self.Y[:int(self.X.shape[0] * train_frac)]
        self.validX = self.X[int(self.X.shape[0] * train_frac):int(self.X.shape[0] * (train_frac + valid_frac))]
        self.validY = self.Y[int(self.X.shape[0] * train_frac):int(self.X.shape[0] * (train_frac + valid_frac))]
        self.testX = self.X[int(self.X.shape[0] * (train_frac + valid_frac)):]
        self.testY = self.Y[int(self.X.shape[0] * (train_frac + valid_frac)):]
        # initial values for solver
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, 2*self.ng:2*self.ng+self.nbus]
        va = Y[:, 2*self.ng+self.nbus:2*self.ng+2*self.nbus]
        return pg, qg, vm, va

    def get_yvars_spv(self, Y):
        pg = Y[:, :self.nspv]
        qg = Y[:, self.nspv:2*self.nspv]
        vm = Y[:, 2*self.nspv:2*self.nspv+self.nbus]
        va = Y[:, 2*self.nspv+self.nbus:2*self.nspv+2*self.nbus]
        return pg, qg, vm, va
    
    def complete_voltage(self, X, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)
        Y_partial[:, :self.ng - 1] = torch.clamp(Z[:, :self.ng - 1], self.pmin[self.non_slack_gen_idx], self.pmax[self.non_slack_gen_idx])
        Y_partial[:, self.ng - 1:] = torch.clamp((self.vmax[self.spv] - self.vmin[self.spv]) * 0.5 * Z[:, self.vm_spv_zidx] + \
            (self.vmax[self.spv] + self.vmin[self.spv]) * 0.5, self.vmin[self.spv], self.vmax[self.spv])
        
        return Y_partial
    # multiple generator buses to the generation bus index (if applicable)
    def gen_map_pv(self):
        gen_bus_unique = torch.tensor(self.spv, dtype=torch.long, device=self.device)
        bus2gen = -torch.ones(self.nbus, dtype=torch.long, device=self.device)
        bus2gen[gen_bus_unique] = torch.arange(len(gen_bus_unique), device=self.device)
        gen_bus_full = torch.tensor(self.ppc['gen'][:, 0] - 1, dtype=int, device=self.device)
        gen_bus_idx = bus2gen[gen_bus_full] 

        slack_mask = gen_bus_full == self.slack[0]
        non_slack_gen_idx = np.where(~slack_mask)[0]
        gen_bus_nonslack = gen_bus_full[non_slack_gen_idx]
        g_slack = np.where(gen_bus_full == self.slack[0])[0][0]

        return gen_bus_idx, gen_bus_unique, gen_bus_full, non_slack_gen_idx, gen_bus_nonslack, g_slack

    def eq_resid_bf(self, X, Y, index = None):
        if index == 1:
            _, _, vm, va = self.get_yvars_spv(Y)
        else:
            _, _, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        vfr = vr@self.Cf.to(self.device) # 8 *41
        vfi = vi@self.Cf.to(self.device) # 8 *41

        ## power balance equations
        tmp1 = vr@torch.transpose(self.Yfbusr.to(self.device),0,1) - vi@torch.transpose(self.Yfbusi.to(self.device),0,1) # batch_size * nbr
        tmp2 = -vr@torch.transpose(self.Yfbusi.to(self.device),0,1) - vi@torch.transpose(self.Yfbusr.to(self.device),0,1)

        # real power flow
        pf = (vfr*tmp1 - vfi*tmp2)

        # reactive power flow
        qf = (vfr*tmp2 + vfi*tmp1)

        return pf, qf

    def get_spv_bounds(self):
        pg_max_spv = torch.zeros(self.nspv, device=self.device, dtype=self.pmax.dtype)
        pg_max_spv.index_add_(dim=0, index=self.gen_bus_idx.long(), source=self.pmax)

        pg_min_spv = torch.zeros(self.nspv, device=self.device, dtype=self.pmin.dtype)
        pg_min_spv.index_add_(0, self.gen_bus_idx.long(), self.pmin)

        qg_min_spv = torch.zeros(self.nspv, device=self.device, dtype=self.qmin.dtype)
        qg_min_spv.index_add_(0, self.gen_bus_idx.long(), self.qmin)

        qg_max_spv = torch.zeros(self.nspv, device=self.device, dtype=self.qmax.dtype)
        qg_max_spv.index_add_(dim=0, index=self.gen_bus_idx.long(), source=self.qmax)
        
        return pg_min_spv, pg_max_spv, qg_min_spv, qg_max_spv
    
    def ineq_violation_loss(self, X, Y, bf, bounded_branch, index = None):
        pg, qg, vm, va = self.get_yvars_spv(Y)
        resids = torch.cat([
            pg - self.pg_max_spv.to(self.device),
            self.pg_min_spv.to(self.device) - pg,
            qg - self.qg_max_spv.to(self.device),
            self.qg_min_spv.to(self.device) - qg,
            vm - self.vmax.to(self.device),
            self.vmin.to(self.device) - vm,
            bf[:, bounded_branch] - self.bfmax[bounded_branch].to(self.device)
        ], dim=1)
        return torch.clamp(resids, 0)

    def gen_cost(self, pg):
        pg_mw = pg * torch.tensor(self.baseMVA, dtype=torch.get_default_dtype(), device=self.device)
        cost = (self.quad_costs * pg_mw**2 + self.lin_costs * pg_mw + self.const_costs).sum(dim=1)
        return cost 
    
    def eq_resid_NR(self, X, Y):
        pg, qg, vm, va = self.get_yvars_spv(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        # power balance equations
        tmp1 = vr@self.Ybusr - vi@self.Ybusi
        tmp2 = -vr@self.Ybusi - vi@self.Ybusr

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        return torch.cat([real_resid, react_resid], dim=1)

    def eq_resid(self, X, Y):
        """Power balance residuals. Generators are aggregated to spv bus level first."""
        pg, qg, vm, va = self.get_yvars(Y)

        # Aggregate generator-level to spv bus level
        pg_spv = torch.zeros(pg.shape[0], self.nspv, device=self.device)
        pg_spv.index_add_(dim=1, index=self.gen_bus_idx, source=pg)

        qg_spv = torch.zeros(pg.shape[0], self.nspv, device=self.device)
        qg_spv.index_add_(dim=1, index=self.gen_bus_idx, source=qg)

        Y_spv = torch.cat([pg_spv, qg_spv, vm, va], dim=1)
        return self.eq_resid_NR(X, Y_spv)
    
# data generating
mpc = scipy.io.loadmat('case_ACTIVSg2000.mat')['ppc']

ppc = case_ACTIVSg2000(mpc)
ppc['gen'][:, 6] = 100
# ALl generators are on
ppc['gen'][:, 7] = 1

mat_data = scipy.io.loadmat('opf_success_genon.mat')

num = 500
Pd = mat_data['Pd_succ'][:, :num]
Qd = mat_data['Qd_succ'][:, :num]
Pg_AC = mat_data['Pg_succ'][:, :num].T
Pg_DC = (np.load("Pg1_DC2.npy").T * 100)[:, :num] # convert to MW

Qg = mat_data['Qg_succ'][:, :num]
Vm = mat_data['Vm_succ'][:, :num]
Va = mat_data['Va_succ'][:, :num]
PFlow = mat_data['PFlow_succ'][:, :num]
QFlow = mat_data['QFlow_succ'][:, :num]
final_objective = mat_data['final_objective_succ'].squeeze()[:num]

# if args['debug']:
#     matpower_data = {}
#     matpower_data["Dem"] = Pd + 1j * Qd
#     matpower_data["Gen"] = Pg_AC.T + 1j * Qg
#     matpower_data["Vol"] = Vm * np.exp(1j * np.radians(Va))
#     data = ACOPFProblem(matpower_data, ppc, num)
#     pf, qf = data.eq_resid_bf(data.X, data.Y, index = 0)
#     pf_diff = PFlow/100 - np.array(pf).T
#     qf_diff = QFlow/100 - np.array(qf).T
#     print(pf_diff.max())
#     print(qf_diff.max())
#     print(pf_diff.min())
#     print(qf_diff.min())
#     res = data.eq_resid(data.X, data.Y)
#     print(res.max())
#     print(res.min())

matpower_data = {}
matpower_data["Dem"] = Pd + 1j * Qd
matpower_data["Gen"] = Pg_DC + 1j * Qg
matpower_data["Vol"] = Vm * np.exp(1j * np.radians(Va))
data = ACOPFProblem(matpower_data, ppc, num)

nbus = data.nbus
bus_indices = list(range(nbus))
gen_buses_indices = data.spv
slack_bus = data.slack
slackva = np.asarray(data.slackva)
Ybusr = scipy.sparse.csr_matrix(np.asarray(data.Ybusr))
Ybusi = scipy.sparse.csr_matrix(np.asarray(data.Ybusi))

pg_max = np.asarray(data.pmax)
pg_min = np.asarray(data.pmin)

qg_max = np.asarray(data.qmax)
qg_min = np.asarray(data.qmin)

pg_bound = np.column_stack((pg_min, pg_max)) 
qg_bound = np.column_stack((qg_min, qg_max))

pv_index = {bus: idx for idx, bus in enumerate(data.pv)} 
slack_index = {bus: idx for idx, bus in enumerate(data.slack)} 
spv_index = {bus: idx for idx, bus in enumerate(data.spv)} 
load_index = {bus: idx for idx, bus in enumerate(data.pq)} 

v_bound = np.column_stack((np.asarray(data.vmin), np.asarray(data.vmax)))

lin_costs = np.asarray(data.lin_costs)
quad_costs = np.asarray(data.quad_costs)
const_costs = np.asarray(data.const_costs)

Ybusr_dict = {
    (int(i), int(j)): Ybusr[i, j]
    for i, j in zip(*Ybusr.nonzero())
}

Ybusi_dict = {
    (int(i), int(j)): Ybusi[i, j]
    for i, j in zip(*Ybusi.nonzero())
}

Ybus_nz = {}
rows, cols = Ybusi.nonzero()
for i, j in zip(rows, cols):
    Ybus_nz.setdefault(i, []).append(j)

from_bus = (ppc['branch'][:, 0] - 1).astype(int)
to_bus = (ppc['branch'][:, 1] - 1).astype(int)
branches = list(zip(from_bus, to_bus))
smax = np.asarray(data.bfmax)

unique_branches = []
unique_smax = []
bounded_branch = []
k = 0
from_bus_all = []
for br, s in zip(branches, smax):
    # if ppc['branch'][k, 5] != 0: for removing the unbounded branch flows 
    unique_branches.append(br)
    unique_smax.append(s)
    bounded_branch.append(k)
    from_bus_all.append(br[0])
    k += 1
# Convert back to np.array if needed
unique_smax = np.array(unique_smax)
Smax_dict = {i: unique_smax[i] for i in range(len(unique_smax))}

Yfbusr = np.asarray(data.Yfbusr)[bounded_branch, :]
Yfbusi = np.asarray(data.Yfbusi)[bounded_branch, :]

# Build dictionary to initialize a Pyomo Param for sparese connection
Yfbusr_dict  = {
    (i, j): Yfbusr[i, j]
    for i in range(Yfbusr.shape[0])
    for j in range(Yfbusr.shape[1])
    if Yfbusr[i, j] != 0.0
}

Yfbusi_dict  = {
    (i, j): Yfbusi[i, j]
    for i in range(Yfbusi.shape[0])
    for j in range(Yfbusi.shape[1])
    if Yfbusi[i, j] != 0.0
}

from_bus_all = np.array(from_bus_all)
from_bus_dict = {i: int(fb) for i, fb in enumerate(from_bus_all)}

Yfbus_nz = {}
rows, cols = (Yfbusr + 1j * Yfbusi).nonzero()  # get row, col indices of nonzero entries
for i, j in zip(rows, cols):
    Yfbus_nz.setdefault(i, []).append(j)

def create_model(nominal_pg, nominal_vg, nominal_pd, nominal_qd):
    # Create a concrete model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define set
    model.buses = pyo.Set(initialize = range(nbus))  # Full bus set
    model.gen_slack_buses = pyo.Set(initialize = data.spv) # Gen bus set
    model.gen_buses = pyo.Set(initialize = data.pv) # Gen bus set
    model.load_buses = pyo.Set(initialize = data.pq)
    model.slack_bus = pyo.Set(initialize = data.slack) # The slack bus
    model.nonslack_idxes = pyo.Set(initialize = data.nonslack_idxes)
    model.Ybus_nz = pyo.Set(model.buses, initialize=lambda m, i: Ybus_nz.get(i, []))
    model.branches = pyo.Set(initialize=unique_branches, dimen=2)
    model.Yfbusr_index = pyo.Set(dimen=2, initialize=Yfbusr_dict.keys())
    model.Yfbusi_index = pyo.Set(dimen=2, initialize=Yfbusi_dict.keys())

    model.all_gens = pyo.Set(initialize=data.non_slack_gen_idx.tolist())
    model.gen_bus = pyo.Param(model.all_gens, initialize={g: int(data.gen_bus_full[g]) for g in data.non_slack_gen_idx}, within=pyo.NonNegativeIntegers)
    model.gens_at_bus = pyo.Set(model.gen_buses, initialize=lambda m, i: [g for g in m.all_gens if m.gen_bus[g] == i])
    
    # Define parameters    
    model.Ybusr = pyo.Param(model.buses, model.buses, initialize=Ybusr_dict, default=0.0)  
    model.Ybusi = pyo.Param(model.buses, model.buses, initialize=Ybusi_dict, default=0.0)

    model.va_slack = pyo.Param(model.slack_bus, initialize = slackva.item())

    model.pd = pyo.Param(model.buses, within = pyo.Reals, initialize = lambda model, i: nominal_pd[i], mutable=True)
    model.qd = pyo.Param(model.buses, within = pyo.Reals, initialize = lambda model, i: nominal_qd[i], mutable=True)

    model.branches_new = pyo.Set(initialize=range(Yfbusr.shape[0]))
    model.from_buses = pyo.Param(model.branches_new, initialize=from_bus_dict)
    model.Yfbusr = pyo.Param(model.branches_new, model.buses, initialize=Yfbusr_dict, default=0.0)
    model.Yfbusi = pyo.Param(model.branches_new, model.buses, initialize=Yfbusi_dict, default=0.0)
    model.Yfbus_nz = pyo.Set(model.branches_new, initialize=lambda m, i: Yfbus_nz.get(i, []))
    model.Smax = pyo.Param(model.branches_new, initialize=Smax_dict)
    model.obj_weight = pyo.Param(initialize=0.0001, within=pyo.NonNegativeReals, mutable=True)
    
    # Define parameters as variables
    # Voltage magnitudes at the generator bus
    model.vg = pyo.Var(model.gen_slack_buses, within=pyo.PositiveReals, initialize=1.0, bounds = lambda model, i: tuple(v_bound[i]))
    model.vg_slack = pyo.Var(model.gen_slack_buses, within=pyo.PositiveReals, initialize=1.0)
    model.pg_all = pyo.Var(model.all_gens, within=pyo.Reals, initialize = 0.0, bounds=lambda model, i: tuple(pg_bound[i]))
    model.pg_all_slack = pyo.Var(model.all_gens, within=pyo.Reals, initialize = 0.0,)

    # Define variables
    # Voltage magnitudes for the load buses
    model.vm = pyo.Var(model.load_buses, within=pyo.PositiveReals, initialize=1.0, bounds = lambda model, i: tuple(v_bound[i])) 
    model.va = pyo.Var(model.nonslack_idxes, within=pyo.Reals, initialize = 0.0)  # Voltage angles at all buses except for the slack bus

    model.pg_ref = pyo.Var(model.slack_bus, within=pyo.Reals, initialize = 0.0, bounds = lambda model, i: tuple(pg_bound[data.g_slack]))
    model.qg_ref = pyo.Var(model.slack_bus, within=pyo.Reals, initialize = 0.0, bounds = lambda model, i: tuple(qg_bound[data.g_slack]))
    model.qg_all = pyo.Var(model.all_gens, within=pyo.Reals, initialize = 0.0, bounds=lambda model, i: tuple(qg_bound[i]))

    model.pg_bus = pyo.Expression(model.gen_buses, rule=lambda m, i: sum(m.pg_all[g] for g in m.gens_at_bus[i]))
    model.qg_bus = pyo.Expression(model.gen_buses, rule=lambda m, i: sum(m.qg_all[g] for g in m.gens_at_bus[i]))

    for i, bus in enumerate(model.all_gens):
        model.pg_all_slack[bus].fix(nominal_pg[i])

    for i, bus in enumerate(model.gen_slack_buses):
        model.vg_slack[bus].fix(nominal_vg[i])

    # PQ bus active power balance
    def active_power_balance(model, i):
        vm_from = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
        va_from = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        if i in model.gen_buses:
            pg = model.pg_bus[i]
        elif i in model.slack_bus:
            pg =  model.pg_ref[i]
        else:
            pg = 0
        ## power balance equations
        inj = 0
        for j in model.Ybus_nz[i]:
            if pyo.value(model.Ybusi[i, j]) != 0:
                vm_to = model.vg[j] if j in model.gen_slack_buses else model.vm[j]
                va_to = model.va[j] if j in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
                inj += vm_from * vm_to * (pyo.cos(va_from - va_to) * model.Ybusr[i, j] + pyo.sin(va_from - va_to) * model.Ybusi[i, j])
        return pg - model.pd[i] == inj

    def reactive_power_balance(model, i):
        vm_from = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
        va_from = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        if i in model.gen_buses:
            qg = model.qg_bus[i]
        elif i in model.slack_bus:
            qg =  model.qg_ref[i]
        else:
            qg = 0
        ## power balance equations
        inj = 0
        for j in model.Ybus_nz[i]:
            if pyo.value(model.Ybusi[i, j]) != 0:
                vm_to = model.vg[j] if j in model.gen_slack_buses else model.vm[j]
                va_to = model.va[j] if j in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
                inj += vm_from * vm_to * (pyo.sin(va_from - va_to) * model.Ybusr[i, j] - pyo.cos(va_from - va_to) * model.Ybusi[i, j])
        return qg - model.qd[i] == inj
    
    def reactive_power_balance_slack(model, i):
        vm_from = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
        va_from = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        if i in model.gen_buses:
            qg = model.qg_bus[i]
        elif i in model.slack_bus:
            qg =  model.qg_ref[i]
        else:
            qg = 0
        ## power balance equations
        inj = 0
        for j in model.Ybus_nz[i]:
            if pyo.value(model.Ybusi[i, j]) != 0:
                vm_to = model.vg[j] if j in model.gen_slack_buses else model.vm[j]
                va_to = model.va[j] if j in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
                inj += vm_from * vm_to * (pyo.sin(va_from - va_to) * model.Ybusr[i, j] - pyo.cos(va_from - va_to) * model.Ybusi[i, j])
        return (qg - model.qd[i]) == inj

    def active_power_balance_slack(model, i):
        vm_from = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
        va_from = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        if i in model.gen_buses:
            pg = model.pg_bus[i]
        elif i in model.slack_bus:
            pg = model.pg_ref[i]
        else:
            pg = 0
        ## power balance equations
        inj = 0
        for j in model.Ybus_nz[i]:
            if pyo.value(model.Ybusi[i, j]) != 0:
                vm_to = model.vg[j] if j in model.gen_slack_buses else model.vm[j]
                va_to = model.va[j] if j in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
                inj += vm_from * vm_to * (pyo.cos(va_from - va_to) * model.Ybusr[i, j] + pyo.sin(va_from - va_to) * model.Ybusi[i, j])
        return (pg - model.pd[i]) == inj
              
    model.active_balance = pyo.Constraint(model.nonslack_idxes, rule=active_power_balance)
    model.active_balance_slack = pyo.Constraint(model.slack_bus, rule=active_power_balance_slack)
    model.reactive_balance = pyo.Constraint(model.nonslack_idxes, rule=reactive_power_balance)
    model.reactive_balance_slack = pyo.Constraint(model.slack_bus, rule=reactive_power_balance_slack)
    
    def calc_Pij(model, f):
        fb = model.from_buses[f]
        vm_from = model.vg[fb] if fb in model.gen_slack_buses else model.vm[fb]
        va_from = model.va[fb] if fb in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        Pij = 0
        for i in model.Yfbus_nz[f]:
            vm_to = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
            va_to = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
            Pij += vm_from * vm_to * (pyo.cos(va_from - va_to) * model.Yfbusr[f, i] + pyo.sin(va_from - va_to) * model.Yfbusi[f, i])
        return Pij

    def calc_Qij(model, f):
        fb = model.from_buses[f]
        vm_from = model.vg[fb] if fb in model.gen_slack_buses else model.vm[fb]
        va_from = model.va[fb] if fb in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
        Qij = 0
        for i in model.Yfbus_nz[f]:
            vm_to = model.vg[i] if i in model.gen_slack_buses else model.vm[i]
            va_to = model.va[i] if i in model.nonslack_idxes else pyo.value(model.va_slack[slack_bus.item()])
            Qij += vm_from * vm_to * (pyo.sin(va_from - va_to) * model.Yfbusr[f, i] - pyo.cos(va_from - va_to) * model.Yfbusi[f, i])
        return Qij
    
    model.Pij = pyo.Expression(model.branches_new, rule = calc_Pij)
    model.Qij = pyo.Expression(model.branches_new, rule = calc_Qij)

    def power_flow_limit(model, f):
        return model.Pij[f]**2 + model.Qij[f]**2 <= model.Smax[f]

    model.PowerFlowLimit = pyo.Constraint(model.branches_new, rule=power_flow_limit)

    # Define the objective function
    def proj_penalty(model):
        return 0.1 * sum((model.pg_all[i] - model.pg_all_slack[i])**2 for i in model.all_gens) +\
               100 * sum((model.vg[i] - model.vg_slack[i])**2 for i in model.gen_slack_buses)
            
    def objective(model):
        slack_cost = sum((quad_costs[data.g_slack] * (model.pg_ref[i] * data.baseMVA)**2 + 
                  lin_costs[data.g_slack] * model.pg_ref[i] * data.baseMVA + 
                          const_costs[data.g_slack]) for i in model.slack_bus)
        
        gen_cost = slack_cost + sum(quad_costs[i] * (model.pg_all[i] * data.baseMVA)**2 + 
                   lin_costs[i] * model.pg_all[i] * data.baseMVA\
                        + const_costs[i] for i in model.all_gens)
        
        return model.obj_weight * gen_cost + proj_penalty(model)

    model.obj = pyo.Objective(expr = objective, sense=pyo.minimize) 
    return model

def _to_numpy_squeezed(x):
    """Convert tensors/arrays to a squeezed NumPy array without deprecated copy behavior."""
    if torch.is_tensor(x):
        return np.squeeze(x.detach().cpu().numpy())
    return np.squeeze(np.asarray(x))


def solve_parallel(grad_inputs, known_inputs, n_workers=2, layer=None, backprop=False):
    """Unified parallel solve helper.

    - Quickstart-style standalone call: `layer=None` (creates context-managed pool).
    - Training-loop call: pass an already-created `layer` to reuse worker pool.
    """
    if layer is None:
        grads = [t.clone().requires_grad_(True) for t in grad_inputs]
        with ParallelPyomoOptLayer(
            model_builder_module='opf_mp_builder',
            model_builder_name='build_dispatch',
            model_builder_args=(),
            var_names=['vm', 'va', 'pg_ref', 'qg_all', 'qg_ref', 'vg', 'pg_all'],
            param_names=['pg_all_slack', 'vg_slack'],
            known_param_names=['pd', 'qd'],
            ipopt_options={'print_level': args['ipopt_print_level']},
            n_workers=n_workers,
        ) as local_layer:
            local_layer.train()
            primal, dual, jac = local_layer(*(grads + list(known_inputs)))
            if backprop:
                primal.sum().backward()
                return primal.detach(), jac.detach(), [t.grad.clone() for t in grads]
            return primal, dual, jac

    primal, dual, jac = layer(*(list(grad_inputs) + list(known_inputs)))
    return primal, dual, jac


class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        self.layers1 = nn.Linear(data.nbus * 2 + data.ng, self._args['hiddenSize'])
        self.relu = nn.ReLU(inplace=True)
        self.layers2 = nn.Linear(self._args['hiddenSize'], data.spv.shape[0] + data.ng - 1)
        self.layers3 = nn.Linear(self._args['hiddenSize'], self._args['hiddenSize'])
        self.layers4 = nn.Linear(self._args['hiddenSize'], self._args['hiddenSize'])

    def forward(self, x):
        out = self.layers1(x)
        out = self.relu(out)

        out = self.layers3(out)
        out = self.relu(out)

        out = self.layers4(out)
        out = self.relu(out)

        out = self.layers2(out)
        out[:, self._data.ng - 1:] = nn.Tanh()(out[:, self._data.ng - 1:])
        return out

class PrimalBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, x, y):
        self.buffer.append((x.detach().clone(), y.detach().clone()))

    def all(self):
        X, Y = zip(*self.buffer)
        return torch.cat(X, dim=0), torch.cat(Y, dim=0)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return sum(x.shape[0] for x, _ in self.buffer)


def main():
    # Fix random seeds for reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sample_index = 0
    nominal_pg, nominal_qg, nominal_vm, nominal_va = data.get_yvars(data.Y[sample_index:sample_index + 1, :])

    nominal_pg = _to_numpy_squeezed(nominal_pg[:, data.non_slack_gen_idx])
    nominal_vg = _to_numpy_squeezed(nominal_vm[:, data.spv])
    nominal_pd = _to_numpy_squeezed(data.X[sample_index, :data.nbus])
    nominal_qd = _to_numpy_squeezed(data.X[sample_index, data.nbus:])
    nominal_vm = _to_numpy_squeezed(nominal_vm[:, data.pq])
    nominal_va = _to_numpy_squeezed(nominal_va[:, data.nonslack_idxes])
    nominal_qg = _to_numpy_squeezed(nominal_qg[:, data.pv_])

    model = create_model(nominal_pg, nominal_vg, nominal_pd, nominal_qd)
    variables_name = [model.vm, model.va, model.pg_ref, model.qg_all, model.qg_ref, model.vg, model.pg_all]
    parameters_name = [model.pg_all_slack, model.vg_slack]
    parameters_known = [model.pd, model.qd]

    if args['num_workers'] > 0:
        if not HAS_MP:
            raise RuntimeError("ParallelPyomoOptLayer is unavailable in this environment.")
        layer_context = ParallelPyomoOptLayer(
            model_builder_module='opf_mp_builder',
            model_builder_name='build_dispatch',
            model_builder_args=(),
            var_names=['vm', 'va', 'pg_ref', 'qg_all', 'qg_ref', 'vg', 'pg_all'],
            param_names=['pg_all_slack', 'vg_slack'],
            known_param_names=['pd', 'qd'],
            ipopt_options={'print_level': args['ipopt_print_level']},
            n_workers=args['num_workers'],
        )
    else:
        serial_layer = PyomoOptLayer(model, variables_name, parameters_name, None, parameters_known)
        layer_context = nullcontext(serial_layer)

    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX, data.trainY)
    valid_dataset = TensorDataset(data.validX, data.validY)
    test_dataset = TensorDataset(data.testX, data.testY)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

    batch_loss = []
    epoch_loss = []
    buffer = PrimalBuffer()

    with layer_context as layer:
        for i in range(nepochs):
            epoch_start = time.perf_counter()
            buffer.clear()
            solver_net.train()
            layer.train()
            for Xtrain, Ytrain in train_loader:
                pg_train, _, _, _ = data.get_yvars(Ytrain)
                pg_train = pg_train.detach()

                solver_opt.zero_grad()

                X_in = torch.cat([Xtrain, pg_train], dim=1)
                Yhat_train_partial = solver_net(X_in)
                Yhat_train_partial = Yhat_train_partial.clone()
                Yhat_train_partial[:, :data.ng - 1] = Yhat_train_partial[:, :data.ng - 1] + X_in[:, data.nbus * 2 + data.non_slack_gen_idx]
                Yhat_train_partial_trans = data.complete_voltage(X_in, Yhat_train_partial).to(torch.float64)

                grad_inputs = [
                    Yhat_train_partial_trans[:, :data.ng - 1],
                    Yhat_train_partial_trans[:, data.ng - 1:],
                ]
                known_inputs = [
                    Xtrain[:, :data.nbus],
                    Xtrain[:, data.nbus:data.nbus * 2],
                ]
                primal_batch, _, _ = solve_parallel(grad_inputs, known_inputs, layer=layer)
                pg_slack = primal_batch[:, -(data.ng - 1):]
                vg_slack = primal_batch[:, len(data.pq) + len(data.nonslack_idxes) + data.ng + 1:len(data.pq) + len(data.nonslack_idxes) + data.ng + 1 + data.spv.shape[0]]
                ## Pyomo output: model.vm, model.va, model.pg_ref, model.qg_all, model.qg_ref, model.vg, model.pg_all
                # Yhat_train = torch.zeros(Xtrain.shape[0], data.ydim, dtype = torch.float32)
                # # pg given by the NN
                # # pg_all:
                # Yhat_train[:, data.non_slack_gen_idx] = pg_slack
                # # pg_ref
                # Yhat_train[:, [data.g_slack]] = primal_batch[:, len(data.pq) + len(data.nonslack_idxes):len(data.pq) + len(data.nonslack_idxes)+1]
                # # qg_all
                # Yhat_train[:, data.ng + data.non_slack_gen_idx] = primal_batch[:, len(data.pq)+len(data.nonslack_idxes)+1:len(data.pq)+len(data.nonslack_idxes)+1+data.ng-1]
                # # qg_ref
                # Yhat_train[:, [data.ng + data.g_slack]] = primal_batch[:, len(data.pq)+len(data.nonslack_idxes)+data.ng:len(data.pq)+len(data.nonslack_idxes)+data.ng+1]
                # # vm 
                # Yhat_train[:, data.ng*2+data.pq] = primal_batch[:, :len(data.pq)]
                # # vg
                # Yhat_train[:, data.ng*2+data.spv] = vg_slack
                # # va
                # Yhat_train[:, data.ng*2+data.nbus+data.nonslack_idxes] = primal_batch[:, len(data.pq):len(data.pq) + len(data.nonslack_idxes)]
                # Yhat_train[:, data.ng*2+data.nbus+data.slack.item()] = slackva.item()
                # calculate the branch flow
                # Pf, Qf = data.eq_resid_bf(Xtrain, Yhat_train, index = 1)
                # sij = torch.square(Pf) + torch.square(Qf)
                pg_slack_ref_diff = torch.norm((Yhat_train_partial_trans[:, :data.ng - 1] - pg_slack), dim=1) + \
                                    100 * torch.norm((Yhat_train_partial_trans[:, data.ng - 1:] - vg_slack), dim=1)
                train_loss = pg_slack_ref_diff.mean()

                train_loss.backward()
                solver_opt.step()
                buffer.add(X_in, torch.cat([pg_slack, vg_slack], dim=1))
                batch_loss.append(train_loss.detach().cpu().numpy())

            X_buf, primal_buf = buffer.all()
            buf_loader = DataLoader(TensorDataset(X_buf, primal_buf), batch_size=batch_size, shuffle=True)
            for _ in range(10):
                for X_in_train, Y_in_train in buf_loader:
                    solver_opt.zero_grad()
                    Yhat_train_partial = solver_net(X_in_train)
                    Yhat_train_partial = Yhat_train_partial.clone()
                    Yhat_train_partial[:, :data.ng - 1] = Yhat_train_partial[:, :data.ng - 1] + X_in_train[:, data.nbus * 2 + data.non_slack_gen_idx]
                    Yhat_train_partial_trans = data.complete_voltage(X_in_train, Yhat_train_partial).to(torch.float64)

                    loss = (torch.norm((Yhat_train_partial_trans[:, :data.ng - 1] - Y_in_train[:, :data.ng - 1]), dim=1) + \
                            100 * torch.norm((Yhat_train_partial_trans[:, data.ng - 1:] - Y_in_train[:, data.ng - 1:]), dim=1)).mean()
                    loss.backward()
                    solver_opt.step()
                    batch_loss.append(loss.detach().cpu().numpy())

            epoch_time = time.perf_counter() - epoch_start
            print('Epoch {}: train loss {:.4f}, epoch time {:.2f}s'.format(i, np.mean(batch_loss), epoch_time), flush=True)
            epoch_loss.append(np.mean(batch_loss))

            # Get valid loss based on PF solver each epoch
            solver_net.eval()
            start_time_eval = time.time()
            with torch.no_grad():
                for Xvalid, Yvalid in valid_loader:
                    pg_valid, _, _, _ = data.get_yvars(Yvalid)
                    pg_valid = pg_valid.detach()
                    X_in = torch.cat([Xvalid, pg_valid], dim=1)
                    Yhat_valid_partial = solver_net(X_in).detach()
                    # add dc pg solution
                    Yhat_valid_partial[:, :data.ng - 1] = Yhat_valid_partial[:, :data.ng - 1] + pg_valid[:, data.non_slack_gen_idx]
                    Yhat_valid_partial_trans = data.complete_voltage(Xvalid, Yhat_valid_partial).to(torch.float64)
                    Yhat_valid = PFFunction_eval(data)(Xvalid, Yhat_valid_partial_trans).to(torch.float64)
                    Pf, Qf = data.eq_resid_bf(Xvalid, Yhat_valid, index = 1)
                    sij_valid = torch.square(Pf) + torch.square(Qf)
            eq_violation = torch.max(data.eq_resid_NR(Xvalid, Yhat_valid), dim=1)[0].detach().cpu().numpy()
            eval_time = time.time() - start_time_eval
            ineq_violation = torch.max(data.ineq_violation_loss(Xvalid, Yhat_valid, sij_valid, bounded_branch), dim=1)[0].detach().cpu().numpy()
            pg_all = torch.zeros(Xvalid.shape[0], data.ng)
            pg_all[:, data.non_slack_gen_idx] = Yhat_valid_partial_trans[:, :data.ng-1]
            pg_all[:, [data.g_slack]] = Yhat_valid[:, data.slack_]
            gen_cost = data.gen_cost(pg_all).detach().cpu().numpy()
            gen_cost_target = data.gen_cost(torch.tensor(Pg_AC[data.trainX.shape[0]:data.trainX.shape[0]+data.validX.shape[0], :]/data.baseMVA, dtype=torch.float64)).detach().cpu().numpy()
            gap = (gen_cost - gen_cost_target) / gen_cost_target * 100
            print('Epoch {}: valid mean eq violation {:.4f}, valid mean ineq violation {:.4f}, mean obj {:.4f} million, gap {:.2f}%, PF eval time {:.2f}s'.format(
                i, np.mean(eq_violation), np.mean(ineq_violation), np.mean(gen_cost) / 1e6, np.mean(gap), eval_time
            ), flush=True)

            batch_loss = []
if __name__ == '__main__' and not IS_POOL_WORKER:
    main()