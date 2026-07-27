###############################################################################
# DC3
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# DC3 also leverages a variety of third-party software packages, which have separate licensing policies.

# This PFFunction was originally part of DC3, available: https://github.com/locuslab/DC3/tree/main
# Copied with modification from https://github.com/locuslab/DC3/blob/main/utils.py
# The reason for using fast decoupled power flow is detailed in the paper ``Unsupervised Deep Learning for AC Optimal Power Flow via Lagrangian Duality"
# Using NR solver also works.
###############################################################################

import torch
import numpy as np
from torch.autograd import Function

def PFFunction_eval(data, tol=1e-5, bsz=50, bsz2=32, max_iters=30):
    class PFFunctionFn_eval(Function):
        @staticmethod
        def forward(ctx, X, Z):
            ## Step 1: Newton's method

            Y = torch.zeros(X.shape[0], data.nspv*2+data.nbus*2, device=data.device, dtype=torch.get_default_dtype())
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            pg = torch.zeros(X.shape[0], data.ng, device=data.device, dtype=torch.get_default_dtype())
            pg[:, data.non_slack_gen_idx] = Z[:, :data.ng-1]  # gen output is from NN
            pg[:, data.g_slack] = 0 # data.Y[:, data.g_slack]
            pg_real = torch.zeros(pg.shape[0], data.nspv, device=data.device) 
            pg_real.index_add_(dim=1, index=data.gen_bus_idx, source=pg)

            # qg = torch.zeros(X.shape[0], data.ng, device=data.device, dtype=torch.get_default_dtype())
            # qg = data.Y[:, data.ng:data.ng*2]
            # qg_real = torch.zeros(qg.shape[0], data.spv.shape[0], device=data.device) 
            # qg_real.index_add_(dim=1, index=data.gen_bus_idx, source=qg)

            Y[:, :data.nspv] = pg_real # generation at non-slack gen
            Y[:, data.nspv:data.nspv*2] = 0 # qg_real # reactive power generation

            Y[:, data.nspv*2 + data.spv] = Z[:, data.ng-1:]   # vm at gens

            # # init guesses for remaining values
            Y[:, data.nspv*2 + data.pq] =  1 # data.Y[:, data.ng*2 + data.pq]  # vm at load buses
            Y[:, data.nspv*2 + data.nbus + data.slack] = torch.tensor(data.slack_va, device=data.device, dtype=torch.get_default_dtype())  # va at slack bus
            Y[:, data.nspv*2 + data.nbus + data.pv] = 0 # data.Y[:, data.ng*2 + data.nbus + data.pv]  # va at non-slack gens 
            Y[:, data.nspv*2 + data.nbus + data.pq] = 0 # data.Y[:, data.ng*2 + data.nbus + data.pq]  # va at load buses

            pg_start_yidx = 0
            qg_start_yidx = data.nspv 
            vm_start_yidx = data.nspv * 2
            va_start_yidx = data.nspv * 2  + data.nbus
            
            # keep_constr = np.concatenate([
            #     data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
            #     data.pflow_start_eqidx + data.pq,     # real power flow at load buses
            #     data.qflow_start_eqidx + data.pq])    # reactive power flow at load buses

            # newton_guess_inds = np.concatenate([             
            #     data.vm_start_yidx + data.pq,         # vm at load buses
            #     data.va_start_yidx + data.pv,         # va at non-slack gens
            #     data.va_start_yidx + data.pq])        # va at load buses
            
            keep_constr1 = np.concatenate([
                data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq])     # real power flow at load buses
            
            keep_constr2 = data.qflow_start_eqidx + data.pq    # reactive power flow at load buses
            
            newton_guess_inds1 = np.concatenate([             
                va_start_yidx + data.pv,         # va at non-slack gens
                va_start_yidx + data.pq])        # va at load buses
            
            newton_guess_inds2 = vm_start_yidx + data.pq         # vm at load buses

            # last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
                
            for b in range(0, X.shape[0], bsz):
                # print('batch: {}'.format(b))
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]
                
                _, _, vm, va = data.get_yvars_spv(Y_b) 

                mis = data.eq_resid_NR(X_b, Y_b) 
                newton_Bp_inv = data.Bp.expand(mis.shape[0], *data.Bp.shape)
                newton_Bpp_inv = data.Bpp.expand(mis.shape[0], *data.Bpp.shape)
                
                gy1 = mis[:, keep_constr1] / vm[:, np.concatenate([data.pv, data.pq])]  # calculate P mismatch
                gy2 = mis[:, keep_constr2] / vm[:, data.pq]  # calculate Q mismatch

                for i in range(max_iters):

                    Va_delta = -newton_Bp_inv.bmm(gy1.unsqueeze(-1)).squeeze(-1)

                    Y_b[:, newton_guess_inds1] -= Va_delta # update the voltage angle 
        
                    _, _, vm, va = data.get_yvars_spv(Y_b) # evalute mismatch
            
                    mis = data.eq_resid_NR(X_b, Y_b) 
                    gy1 = mis[:, keep_constr1] / vm[:, np.concatenate([data.pv, data.pq])]  # calculate P mismatch
                    gy2 = mis[:, keep_constr2] / vm[:, data.pq]  # calculate Q mismatch
                    
                    if torch.norm(gy1, dim=1).abs().max() < tol and torch.norm(gy2, dim=1).abs().max() < tol:
                        break

                    Vm_delta = -newton_Bpp_inv.bmm(gy2.unsqueeze(-1)).squeeze(-1)

                    Y_b[:, newton_guess_inds2] -= Vm_delta  # update the voltage magnitude
                    
                    _, _, vm, va = data.get_yvars_spv(Y_b) # evalute mismatch
                    mis = data.eq_resid_NR(X_b, Y_b)
                    
                    gy1 = mis[:, keep_constr1] / vm[:, np.concatenate([data.pv, data.pq])]  # calculate P mismatch
                    gy2 = mis[:, keep_constr2] / vm[:, data.pq]  # calculate Q mismatch

                    if torch.norm(gy1, dim=1).abs().max() < tol and torch.norm(gy2, dim=1).abs().max() < tol:
                        break
                
                Y_b[:, qg_start_yidx:qg_start_yidx + data.nspv] = -mis[:, data.qflow_start_eqidx + data.spv]

                Y_b[:,pg_start_yidx + data.slack_] = -mis[:, data.pflow_start_eqidx + data.slack]
            
            return Y

    return PFFunctionFn_eval.apply