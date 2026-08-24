"""Dynamic-sparse LTC cell."""

from dataclasses import replace
from typing import Optional, Tuple, Union

import ncps
import torch
from ncps.torch import LTCCell

from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import Router, RouterConfig, RouterOutput


class DSLTCCell(LTCCell):
    """LTC cell with input-dependent sparse gating.

    Inherits the ODE-solver machinery from ncps.torch.LTCCell and adds channel
    masks from two routers (input and hidden channels). The masks scale the
    per-connection activations inside the ODE solver: the hidden mask on the
    pre-synaptic channels, the input mask on the sensory channels.
    """

    supports_precompute: bool = True

    def __init__(self,
                 input_size: int,
                 hidden_size: Union[int, ncps.wirings.Wiring],
                 router_config: RouterConfig,
                 implementation: Implementation = "dense",
                 share_context_projection: bool = False,
                 input_mapping: str = "affine",
                 output_mapping: str = "identity",
                 ode_unfolds: int = 6,
                 epsilon: float = 1e-8,
                 implicit_param_constraints: bool = True,
                 use_sensor_dynamic_mask: bool = True):
        """Initialize the cell.

        Args:
            input_size: Number of input channels.
            hidden_size: Number of LTC units, or an ncps Wiring. If an int, a
                fully-connected wiring is used.
            router_config: Configuration for the hidden-channel router. The
                input-channel router is derived from it by setting
                num_channels=input_size.
            implementation: Masked matmul implementation. Only "dense" is
                currently supported.
            share_context_projection: If True, the cell owns a single context
                projection shared by both routers. Requires
                router_config.context_dim.
            input_mapping: LTC input mapping type.
            output_mapping: LTC output mapping type. "identity" returns the
                raw state, leaving the readout to the model's projection.
            ode_unfolds: Number of ODE solver unrolls per step.
            epsilon: Numerical stability constant.
            implicit_param_constraints: Whether to constrain parameters
                implicitly.
            use_sensor_dynamic_mask: Whether to apply the input mask to the
                sensory pathway.
        """
        if implementation != "dense":
            raise NotImplementedError(
                f"DSLTCCell currently supports only the dense implementation, got '{implementation}'.")

        if isinstance(hidden_size, ncps.wirings.Wiring):
            wiring = hidden_size
            units = wiring.units
        else:
            units = hidden_size
            wiring = ncps.wirings.FullyConnected(units)

        super().__init__(wiring=wiring,
                         in_features=input_size,
                         input_mapping=input_mapping,
                         output_mapping=output_mapping,
                         ode_unfolds=ode_unfolds,
                         epsilon=epsilon,
                         implicit_param_constraints=implicit_param_constraints)

        if router_config.input_dim != input_size:
            raise ValueError(f"router_config.input_dim ({router_config.input_dim}) must equal "
                             f"input_size ({input_size}).")
        if router_config.num_channels != self.state_size:
            raise ValueError(f"router_config.num_channels ({router_config.num_channels}) must equal "
                             f"the number of units ({self.state_size}).")

        self.implementation = implementation
        self.share_context_projection = share_context_projection
        self.use_sensor_dynamic_mask = use_sensor_dynamic_mask

        if share_context_projection:
            if router_config.context_dim is None:
                raise ValueError("share_context_projection requires router_config.context_dim.")
            self.context_proj: Optional[torch.nn.Module] = torch.nn.Linear(input_size,
                                                                           router_config.context_dim)
            base_config = replace(router_config, input_dim=router_config.context_dim, context_dim=None)
        else:
            self.context_proj = None
            base_config = router_config

        self.hidden_router = Router(base_config)
        self.input_router = Router(replace(base_config, num_channels=input_size))

        self.last_input_routing: Optional[RouterOutput] = None
        self.last_hidden_routing: Optional[RouterOutput] = None

    def compute_routing(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute input and hidden channel masks for the current input.

        Args:
            x: Current cell input of shape (B, input_size), after the LTC
                input mapping.

        Returns:
            The (input_routing, hidden_routing) pair.
        """
        context = self.context_proj(x) if self.context_proj is not None else x
        self.last_input_routing = self.input_router(context)
        self.last_hidden_routing = self.hidden_router(context)
        return self.last_input_routing, self.last_hidden_routing

    def compute_routing_batched(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute routing for a flattened batch of raw inputs.

        Args:
            x: Flattened raw inputs of shape (N, input_size), e.g. N = B * T.
                The LTC input mapping is applied before routing.

        Returns:
            The (input_routing, hidden_routing) pair over all N inputs.
        """
        mapped = self._map_inputs(x)
        context = self.context_proj(mapped) if self.context_proj is not None else mapped
        input_routing = self.input_router(context)
        hidden_routing = self.hidden_router(context)
        return input_routing, hidden_routing

    def _ode_solver(self,
                    inputs: torch.Tensor,
                    state: torch.Tensor,
                    elapsed_time: float,
                    input_routing: RouterOutput,
                    hidden_routing: RouterOutput) -> torch.Tensor:
        sensory_w_activation = self.make_positive_fn(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"])
        if self.use_sensor_dynamic_mask:
            sensory_w_activation = sensory_w_activation * input_routing.mask.unsqueeze(-1)
        sensory_w_activation = sensory_w_activation * self._params["sensory_sparsity_mask"]
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        cm_t = self.make_positive_fn(self._params["cm"]) / (elapsed_time / self._ode_unfolds)
        w_param = self.make_positive_fn(self._params["w"])
        v_pre = state
        for _ in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            w_activation = w_activation * self._params["sparsity_mask"] * hidden_routing.mask.unsqueeze(-1)
            rev_activation = w_activation * self._params["erev"]

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator
            v_pre = numerator / (denominator + self._epsilon)
        return v_pre

    def forward(self,
                inputs: torch.Tensor,
                states: torch.Tensor,
                elapsed_time: float = 1.0,
                routing: Optional[Tuple[RouterOutput, RouterOutput]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one cell step.

        Args:
            inputs: Current input of shape (B, input_size).
            states: Current state of shape (B, state_size).
            elapsed_time: Elapsed time since the last update.
            routing: Optional precomputed (input_routing, hidden_routing) pair.
                If None, routing is computed from the mapped input.

        Returns:
            A (output, next_state) pair. output has shape (B, state_size) when
            output_mapping="identity".
        """
        mapped_inputs = self._map_inputs(inputs)
        if routing is None:
            input_routing, hidden_routing = self.compute_routing(mapped_inputs)
        else:
            input_routing, hidden_routing = routing
            self.last_input_routing = input_routing
            self.last_hidden_routing = hidden_routing
        next_state = self._ode_solver(mapped_inputs, states, elapsed_time, input_routing,
                                      hidden_routing)
        outputs = self._map_outputs(next_state)
        return outputs, next_state

    def reset_parameters(self) -> None:
        """Reset the routers and the shared context projection, if present."""
        self.input_router.reset_parameters()
        self.hidden_router.reset_parameters()
        if self.context_proj is not None:
            self.context_proj.reset_parameters()

    def extra_repr(self) -> str:
        return (f"input_size={self.input_size}, state_size={self.state_size}, "
                f"implementation='{self.implementation}', "
                f"share_context_projection={self.share_context_projection}, "
                f"use_sensor_dynamic_mask={self.use_sensor_dynamic_mask}")
