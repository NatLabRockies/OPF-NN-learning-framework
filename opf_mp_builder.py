import os
import sys
from copy import deepcopy

import numpy as np
import pyomo.environ as pyo
import scipy
import torch
from pypower.api import makeYbus
from pypower import idx_bus, idx_gen

# Keep local imports stable when workers spawn from different cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from case_ACTIVSg2000 import case_ACTIVSg2000

# Keep dtype aligned with OPF scripts.
torch.set_default_dtype(torch.float64)

# Compatibility shim for older PYPOWER expectations.
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan


class ACOPFProblem:
    def __init__(self, data, ppc, num):
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
        self.gen_bus_idx, self.gen_bus_unique, self.gen_bus_full, self.non_slack_gen_idx, self.gen_bus_nonslack, self.g_slack = self.gen_map_pv()

        self.ng = ppc['gen'].shape[0]
        self.quad_costs = torch.tensor(ppc['gencost'][:, 4], dtype=torch.get_default_dtype())
        self.lin_costs = torch.tensor(ppc['gencost'][:, 5], dtype=torch.get_default_dtype())
        self.const_costs = torch.tensor(ppc['gencost'][:, 6], dtype=torch.get_default_dtype())

        self.pmax = torch.tensor(ppc['gen'][:, idx_gen.PMAX] / self.baseMVA, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:, idx_gen.PMIN] / self.baseMVA, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:, idx_gen.QMAX] / self.baseMVA, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:, idx_gen.QMIN] / self.baseMVA, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:, idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:, idx_bus.VMIN], dtype=torch.get_default_dtype())
        slackva_np = np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])
        self.slackva = torch.as_tensor(slackva_np, dtype=torch.get_default_dtype())
        self.bfmax = torch.tensor((ppc['branch'][:, 5] / self.baseMVA) ** 2, dtype=torch.get_default_dtype())

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

        demand = data['Dem'].T / self.baseMVA
        gen = data['Gen'].T / self.baseMVA
        voltage = data['Vol'].T

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)[:num, :]
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(voltage), np.angle(voltage)], axis=1)[:num, :]

        self.X = torch.tensor(X, dtype=torch.get_default_dtype())
        self.Y = torch.tensor(Y, dtype=torch.get_default_dtype())

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2 * self.ng]
        vm = Y[:, 2 * self.ng:2 * self.ng + self.nbus]
        va = Y[:, 2 * self.ng + self.nbus:2 * self.ng + 2 * self.nbus]
        return pg, qg, vm, va

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


def _to_numpy_squeezed(x):
    if torch.is_tensor(x):
        return np.squeeze(x.detach().cpu().numpy())
    return np.squeeze(np.asarray(x))


_CONTEXT = None


def _ensure_context():
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT

    mpc = scipy.io.loadmat(os.path.join(_HERE, 'case_ACTIVSg2000.mat'))['ppc']
    ppc = case_ACTIVSg2000(mpc)
    ppc['gen'][:, 6] = 100
    ppc['gen'][:, 7] = 1

    mat_data = scipy.io.loadmat(os.path.join(_HERE, 'opf_success_genon.mat'))
    num = 10
    Pd = mat_data['Pd_succ'][:, :num]
    Qd = mat_data['Qd_succ'][:, :num]
    Pg_DC = np.load(os.path.join(_HERE, 'Pg1_DC2.npy')).T * 100
    Qg = mat_data['Qg_succ'][:, :num]
    Vm = mat_data['Vm_succ'][:, :num]
    Va = mat_data['Va_succ'][:, :num]

    matpower_data = {
        'Dem': Pd + 1j * Qd,
        'Gen': Pg_DC[:, :num] + 1j * Qg,
        'Vol': Vm * np.exp(1j * np.radians(Va)),
    }

    data = ACOPFProblem(matpower_data, ppc, num)
    nbus = data.nbus
    slack_bus = data.slack
    slackva = np.asarray(data.slackva)

    Ybusr = scipy.sparse.csr_matrix(np.asarray(data.Ybusr))
    Ybusi = scipy.sparse.csr_matrix(np.asarray(data.Ybusi))
    Ybusr_dict = {(int(i), int(j)): Ybusr[i, j] for i, j in zip(*Ybusr.nonzero())}
    Ybusi_dict = {(int(i), int(j)): Ybusi[i, j] for i, j in zip(*Ybusi.nonzero())}
    Ybus_nz = {}
    rows, cols = Ybusi.nonzero()
    for i, j in zip(rows, cols):
        Ybus_nz.setdefault(i, []).append(j)

    pg_bound = np.column_stack((np.asarray(data.pmin), np.asarray(data.pmax)))
    qg_bound = np.column_stack((np.asarray(data.qmin), np.asarray(data.qmax)))
    v_bound = np.column_stack((np.asarray(data.vmin), np.asarray(data.vmax)))

    from_bus = (ppc['branch'][:, 0] - 1).astype(int)
    branches = list(zip(from_bus, (ppc['branch'][:, 1] - 1).astype(int)))
    smax = np.asarray(data.bfmax)

    bounded_branch = list(range(len(branches)))
    unique_branches = list(branches)
    unique_smax = np.array([s for s in smax])
    Smax_dict = {i: unique_smax[i] for i in range(len(unique_smax))}

    Yfbusr = np.asarray(data.Yfbusr)[bounded_branch, :]
    Yfbusi = np.asarray(data.Yfbusi)[bounded_branch, :]

    Yfbusr_dict = {
        (i, j): Yfbusr[i, j]
        for i in range(Yfbusr.shape[0])
        for j in range(Yfbusr.shape[1])
        if Yfbusr[i, j] != 0.0
    }
    Yfbusi_dict = {
        (i, j): Yfbusi[i, j]
        for i in range(Yfbusi.shape[0])
        for j in range(Yfbusi.shape[1])
        if Yfbusi[i, j] != 0.0
    }

    from_bus_dict = {i: int(fb) for i, fb in enumerate(np.array([br[0] for br in branches]))}
    Yfbus_nz = {}
    rows, cols = (Yfbusr + 1j * Yfbusi).nonzero()
    for i, j in zip(rows, cols):
        Yfbus_nz.setdefault(i, []).append(j)

    lin_costs = np.asarray(data.lin_costs)
    quad_costs = np.asarray(data.quad_costs)
    const_costs = np.asarray(data.const_costs)

    sample_index = 0
    nominal_pg, nominal_qg, nominal_vm, nominal_va = data.get_yvars(data.Y[sample_index:sample_index + 1, :])
    nominal_pg = _to_numpy_squeezed(nominal_pg[:, data.non_slack_gen_idx])
    nominal_vg = _to_numpy_squeezed(nominal_vm[:, data.spv])
    nominal_pd = _to_numpy_squeezed(data.X[sample_index, :data.nbus])
    nominal_qd = _to_numpy_squeezed(data.X[sample_index, data.nbus:])

    _CONTEXT = {
        'data': data,
        'nbus': nbus,
        'slack_bus': slack_bus,
        'slackva': slackva,
        'Ybusr_dict': Ybusr_dict,
        'Ybusi_dict': Ybusi_dict,
        'Ybus_nz': Ybus_nz,
        'pg_bound': pg_bound,
        'qg_bound': qg_bound,
        'v_bound': v_bound,
        'unique_branches': unique_branches,
        'Yfbusr': Yfbusr,
        'Yfbusi': Yfbusi,
        'Yfbusr_dict': Yfbusr_dict,
        'Yfbusi_dict': Yfbusi_dict,
        'from_bus_dict': from_bus_dict,
        'Yfbus_nz': Yfbus_nz,
        'Smax_dict': Smax_dict,
        'lin_costs': lin_costs,
        'quad_costs': quad_costs,
        'const_costs': const_costs,
        'nominal_pg': nominal_pg,
        'nominal_vg': nominal_vg,
        'nominal_pd': nominal_pd,
        'nominal_qd': nominal_qd,
    }
    return _CONTEXT


def create_model(nominal_pg, nominal_vg, nominal_pd, nominal_qd):
    ctx = _ensure_context()
    data = ctx['data']
    nbus = ctx['nbus']
    slack_bus = ctx['slack_bus']
    slackva = ctx['slackva']
    Ybusr_dict = ctx['Ybusr_dict']
    Ybusi_dict = ctx['Ybusi_dict']
    Ybus_nz = ctx['Ybus_nz']
    pg_bound = ctx['pg_bound']
    qg_bound = ctx['qg_bound']
    v_bound = ctx['v_bound']
    unique_branches = ctx['unique_branches']
    Yfbusr = ctx['Yfbusr']
    Yfbusi = ctx['Yfbusi']
    Yfbusr_dict = ctx['Yfbusr_dict']
    Yfbusi_dict = ctx['Yfbusi_dict']
    from_bus_dict = ctx['from_bus_dict']
    Yfbus_nz = ctx['Yfbus_nz']
    Smax_dict = ctx['Smax_dict']
    lin_costs = ctx['lin_costs']
    quad_costs = ctx['quad_costs']
    const_costs = ctx['const_costs']

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

def build_dispatch():
    ctx = _ensure_context()
    model = create_model(ctx['nominal_pg'], ctx['nominal_vg'], ctx['nominal_pd'], ctx['nominal_qd'])
    var_map = {
        'vm': model.vm,
        'va': model.va,
        'pg_ref': model.pg_ref,
        'qg_all': model.qg_all,
        'qg_ref': model.qg_ref,
        'vg': model.vg,
        'pg_all': model.pg_all,
    }
    param_map = {
        'pg_all_slack': model.pg_all_slack,
        'vg_slack': model.vg_slack,
    }
    known_param_map = {
        'pd': model.pd,
        'qd': model.qd,
    }
    return model, var_map, param_map, known_param_map
