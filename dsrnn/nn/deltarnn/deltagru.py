"""Delta-based GRU (DeltaGRU), Neil et al. 2017.

Faithful port of ``connectomencps.nn.deltarnn.deltagru``; the import paths
changed (``dsrnn.nn.activations`` / ``dsrnn.nn.quantization``), and the
``@torch.jit.script`` decorators on the delta/gate helpers were removed —
TorchScript scripting is incompatible with the ``inference_mode`` Lightning
uses for evaluation (it raised "Inference tensors cannot be saved for
backward" during ``trainer.test()``). The helpers are numerically identical as
plain functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor

from dsrnn.nn.activations import hardsigmoid
from dsrnn.nn.quantization import quantize_tensor


class DeltaGRU(nn.GRU):
    def __init__(self,
                 input_size=16,
                 hidden_size=256,
                 num_layers=2,
                 thx=0,
                 thh=0,
                 qa=0,
                 aqi=8,
                 aqf=8,
                 qw=0,
                 wqi=1,
                 wqf=7,
                 nqi=2,
                 nqf=4,
                 bw_acc=32,
                 use_hardsigmoid=0,
                 use_hardtanh=0,
                 debug=0):
        """Delta-based GRU: operates on input/hidden deltas, suppressing
        sub-threshold changes (temporal sparsity); optional fixed-point
        quantization and hard nonlinearities."""
        super(DeltaGRU, self).__init__(input_size, hidden_size, num_layers)

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.th_x: float = thx
        self.th_h: float = thh
        self.qa: int = qa
        self.aqi: int = aqi
        self.aqf: int = aqf
        self.qw: int = qw
        self.wqi: int = wqi
        self.wqf: int = wqf
        self.nqi: int = nqi
        self.nqf: int = nqf
        self.bw_acc: int = bw_acc
        self.debug: int = debug
        self.weight_ih_height: int = 3 * self.hidden_size
        self.weight_ih_width: int = self.input_size
        self.weight_hh_width: int = self.hidden_size
        self.use_hardsigmoid: int = use_hardsigmoid
        self.use_hardtanh: int = use_hardtanh
        self.x_p_length: int = max(self.input_size, self.hidden_size)

        self.abs_sum_delta_hid = torch.zeros(1)
        self.sp_dx = 0
        self.sp_dh = 0

        self.statistics = None
        self.set_debug(self.debug)

    def set_debug(self, value):
        setattr(self, "debug", value)
        self.statistics = {
            "num_dx_zeros": 0,
            "num_dx_numel": 0,
            "num_dh_zeros": 0,
            "num_dh_numel": 0,
        }

    def add_to_debug(self, x, i_layer, name):
        if self.debug:
            if isinstance(x, Tensor):
                variable = np.squeeze(x.cpu().numpy())
            else:
                variable = np.squeeze(np.asarray(x))
            variable_name = "_".join(["l" + str(i_layer), name])
            if variable_name not in self.statistics.keys():
                self.statistics[variable_name] = []
            self.statistics[variable_name].append(variable)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "l0" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param[:self.hidden_size, :])
                    nn.init.xavier_uniform_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.xavier_uniform_(param[2 * self.hidden_size:3 * self.hidden_size, :])
                if "weight_hh" in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
            else:
                if "weight" in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
            if "bias" in name:
                nn.init.constant_(param, 0)

    def get_temporal_sparsity(self):
        temporal_sparsity = {}
        if self.debug:
            temporal_sparsity["SP_T_DX"] = float(self.statistics["num_dx_zeros"] / self.statistics["num_dx_numel"])
            temporal_sparsity["SP_T_DH"] = float(self.statistics["num_dh_zeros"] / self.statistics["num_dh_numel"])
            temporal_sparsity["SP_T_DV"] = float((self.statistics["num_dx_zeros"] + self.statistics["num_dh_zeros"]) /
                                                 (self.statistics["num_dx_numel"] + self.statistics["num_dh_numel"]))
        self.statistics.update(temporal_sparsity)
        return temporal_sparsity

    def process_inputs(self, x: Tensor, qa: int, x_p_0: Tensor = None, h_0: Tensor = None,
                       h_p_0: Tensor = None, dm_ch_0: Tensor = None, dm_0: Tensor = None):
        if self.batch_first:
            x.transpose(0, 1)
            setattr(self, "batch_size", int(x.size()[0]))
        else:
            setattr(self, "batch_size", int(x.size()[1]))
        batch_size = x.size()[1]
        x = quantize_tensor(x, self.aqi, self.aqf, qa)

        if x_p_0 is None or h_0 is None or h_p_0 is None or dm_ch_0 is None or dm_0 is None:
            x_p_0 = torch.zeros(self.num_layers, batch_size, self.x_p_length,
                                dtype=x.dtype, device=x.device)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            h_p_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_ch_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_0 = torch.zeros(self.num_layers, batch_size, self.weight_ih_height, dtype=x.dtype, device=x.device)
            for l in range(self.num_layers):
                bias_ih = getattr(self, "bias_ih_l{}".format(l))
                bias_hh = getattr(self, "bias_hh_l{}".format(l))
                dm_0[l, :, :self.hidden_size] = quantize_tensor(dm_0[l, :, :self.hidden_size] +
                                                                bias_ih[:self.hidden_size] +
                                                                bias_hh[:self.hidden_size], self.wqi, self.wqf, self.qw)
                dm_0[l, :, self.hidden_size:2 * self.hidden_size] = quantize_tensor(
                    dm_0[l, :, self.hidden_size:2 * self.hidden_size] +
                    bias_ih[self.hidden_size:2 * self.hidden_size] +
                    bias_hh[self.hidden_size:2 * self.hidden_size], self.wqi, self.wqf, self.qw)
                dm_0[l, :, 2 * self.hidden_size:3 * self.hidden_size] = quantize_tensor(
                    dm_0[l, :, 2 * self.hidden_size:3 * self.hidden_size] +
                    bias_ih[2 * self.hidden_size:3 * self.hidden_size], self.wqi,
                    self.wqf, self.qw)
                dm_ch_0[l, :, :] = quantize_tensor(
                    dm_ch_0[l, :, :] + bias_hh[2 * self.hidden_size:3 * self.hidden_size],
                    self.wqi, self.wqf, self.qw)

        return x, x_p_0, h_0, h_p_0, dm_ch_0, dm_0

    @staticmethod
    def compute_deltas(x: Tensor, x_p: Tensor, h: Tensor, h_p: Tensor, th_x: Tensor, th_h: Tensor):
        delta_x = x - x_p
        delta_h = h - h_p
        delta_x_abs = torch.abs(delta_x)
        delta_x = delta_x.masked_fill(delta_x_abs < th_x, 0)
        delta_h_abs = torch.abs(delta_h)
        delta_h = delta_h.masked_fill(delta_h_abs < th_h, 0)
        return delta_x, delta_h, delta_x_abs, delta_h_abs

    @staticmethod
    def update_states(delta_x_abs, delta_h_abs, x, h, x_p, h_p, x_prev_out, th_x, th_h):
        x_p = torch.where(delta_x_abs >= th_x, x, x_p)
        x_prev_out[:, :x.size(-1)] = x_p
        h_p = torch.where(delta_h_abs >= th_h, h, h_p)
        return x_p, h_p, x_prev_out

    @staticmethod
    def compute_gates(delta_x: Tensor, delta_h: Tensor, dm: Tensor, dm_nh: Tensor,
                      weight_ih: Tensor, weight_hh: Tensor):
        mac_x = torch.mm(delta_x, weight_ih.t()) + dm
        mac_h = torch.mm(delta_h, weight_hh.t())
        mac_x_chunks = mac_x.chunk(3, dim=1)
        mac_h_chunks = mac_h.chunk(3, dim=1)
        dm_r = mac_x_chunks[0] + mac_h_chunks[0]
        dm_z = mac_x_chunks[1] + mac_h_chunks[1]
        dm_n = mac_x_chunks[2]
        dm_nh = mac_h_chunks[2] + dm_nh
        dm = torch.cat((dm_r, dm_z, dm_n), 1)
        return dm, dm_r, dm_z, dm_n, dm_nh

    def layer_forward(self, input: Tensor, l: int, qa: int, x_p_0: Tensor = None,
                      h_0: Tensor = None, h_p_0: Tensor = None, dm_nh_0: Tensor = None,
                      dm_0: Tensor = None):
        weight_ih = getattr(self, "weight_ih_l{}".format(l))
        weight_hh = getattr(self, "weight_hh_l{}".format(l))

        input_size = input.size(-1)
        batch_size = input.size(1)

        th_x = quantize_tensor(torch.tensor(self.th_x, dtype=input.dtype), self.aqi, self.aqf, qa)
        th_h = quantize_tensor(torch.tensor(self.th_h, dtype=input.dtype), self.aqi, self.aqf, qa)

        inputs = quantize_tensor(input, self.aqi, self.aqf, qa)
        inputs = inputs.unbind(0)

        output = []
        reg = torch.zeros(1, dtype=input.dtype, device=input.device).squeeze()

        x_p_out = torch.zeros(batch_size, self.x_p_length, dtype=input.dtype, device=input.device)
        x_p = quantize_tensor(x_p_0[:, :input_size], self.aqi, self.aqf, qa)
        x_prev_out_size = torch.zeros_like(x_p)
        x_prev_out = quantize_tensor(x_prev_out_size, self.aqi, self.aqf, qa)
        h = quantize_tensor(h_0, self.aqi, self.aqf, qa)
        h_p = quantize_tensor(h_p_0, self.aqi, self.aqf, qa)
        dm_nh = quantize_tensor(dm_nh_0, self.aqi, self.aqf, qa)
        dm = dm_0

        seq_len = len(inputs)
        for t in range(seq_len):
            x = inputs[t]
            delta_x, delta_h, delta_x_abs, delta_h_abs = self.compute_deltas(x, x_p, h, h_p, th_x, th_h)
            reg += torch.sum(torch.abs(delta_h))

            if self.debug:
                zero_mask_delta_x = torch.as_tensor(delta_x == 0, dtype=x.dtype)
                zero_mask_delta_h = torch.as_tensor(delta_h == 0, dtype=x.dtype)
                self.statistics["num_dx_zeros"] += torch.sum(zero_mask_delta_x)
                self.statistics["num_dh_zeros"] += torch.sum(zero_mask_delta_h)
                self.statistics["num_dx_numel"] += torch.numel(delta_x)
                self.statistics["num_dh_numel"] += torch.numel(delta_h)

            x_p, h_p, x_prev_out = self.update_states(delta_x_abs, delta_h_abs, x, h, x_p, h_p,
                                                      x_prev_out, th_x, th_h)

            dm, dm_r, dm_z, dm_n, dm_nh = self.compute_gates(delta_x, delta_h, dm, dm_nh,
                                                             weight_ih, weight_hh)

            pre_act_r = quantize_tensor(dm_r, self.aqi, self.aqf, qa)
            pre_act_z = quantize_tensor(dm_z, self.aqi, self.aqf, qa)

            gate_r = hardsigmoid(pre_act_r) if self.use_hardsigmoid else torch.sigmoid(pre_act_r)
            gate_z = hardsigmoid(pre_act_z) if self.use_hardsigmoid else torch.sigmoid(pre_act_z)
            q_r = quantize_tensor(gate_r, self.nqi, self.nqf, qa)
            q_z = quantize_tensor(gate_z, self.nqi, self.nqf, qa)

            pre_act_nh = quantize_tensor(dm_nh, self.aqi, self.aqf, qa)
            pre_act_n = quantize_tensor(dm_n + torch.mul(q_r, pre_act_nh), self.aqi, self.aqf, qa)
            gate_n = nnf.hardtanh(pre_act_n) if self.use_hardtanh else torch.tanh(pre_act_n)
            q_n = quantize_tensor(gate_n, self.nqi, self.nqf, qa)

            one_minus_u = torch.ones_like(q_z, device=q_z.device) - q_z
            a = quantize_tensor(torch.mul(one_minus_u, q_n), self.aqi, self.aqf, qa)
            b = quantize_tensor(torch.mul(q_z, h), self.aqi, self.aqf, self.qa)
            h = a + b
            h = quantize_tensor(h, self.aqi, self.aqf, qa)

            output += [h]

        output = torch.stack(output)
        x_p_out[:, :input_size] = x_p
        return output, (x_p_out, h, h_p, dm_nh, dm), reg

    def forward(self, input: Tensor, x_p_0: Tensor = None, h_0: Tensor = None,
                h_p_0: Tensor = None, dm_nh_0: Tensor = None, dm_0: Tensor = None):
        qa = 0 if self.training else self.qa

        x, x_p_0, h_0, h_p_0, dm_nh_0, dm_0 = self.process_inputs(
            input, qa, x_p_0, h_0, h_p_0, dm_nh_0, dm_0)

        reg = torch.zeros(1, dtype=x.dtype, device=x.device).squeeze()
        x_p_n, h_n, h_p_n, dm_nm_n, dm_n = [], [], [], [], []
        for l in range(self.num_layers):
            x, (x_p_n_l, h_n_l, h_p_n_l, dm_nh_n_l, dm_n_l), reg_l = self.layer_forward(
                x, l, qa, x_p_0[l], h_0[l], h_p_0[l], dm_nh_0[l], dm_0[l])
            x_p_n.append(x_p_n_l)
            h_n.append(h_n_l)
            h_p_n.append(h_p_n_l)
            dm_nm_n.append(dm_nh_n_l)
            dm_n.append(dm_n_l)
            reg += reg_l
        x_p_n = torch.stack(x_p_n)
        h_n = torch.stack(h_n)
        h_p_n = torch.stack(h_p_n)
        dm_nm_n = torch.stack(dm_nm_n)
        dm_n = torch.stack(dm_n)

        if self.debug:
            self.statistics["sparsity_dx"] = float(self.statistics["num_dx_zeros"] / self.statistics["num_dx_numel"])
            self.statistics["sparsity_dh"] = float(self.statistics["num_dh_zeros"] / self.statistics["num_dh_numel"])
            self.statistics["sparsity_to"] = float((self.statistics["num_dx_zeros"] + self.statistics["num_dh_zeros"]) /
                                                   (self.statistics["num_dx_numel"] + self.statistics["num_dh_numel"]))

        return x, (x_p_n, h_n, h_p_n, dm_nm_n, dm_n), reg
