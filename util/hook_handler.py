import torch
import numpy as np
from typing import Iterable, Optional, Dict, Any, Callable, Tuple, List, Set
import argparse
import logging

general_hook_handler_built = False

# inputs and outputs: coefficient of variation and L2 norm
# weights: induced 2-norm and stable rank for matrices, cv and L2 for biases and kernels
# gradients: relative norm and cosine similarity with respect to the weight matrix

class GeneralHookHandler:

    def __init__(
        self,
        model: torch.nn.Module,
        tracked_module_names: Iterable[str],
        tracking_interval: int,
        general_logger: Any,
        weight_logger: Any,
        input_output_logger: Any,
        gradient_logger: Any,
        writer: Any,
        device_cpu_offload: bool,
        enabled: bool
    ):
        self.model = model
        self.tracked_module_names: Set[str] = set(tracked_module_names)
        self.tracking_interval = max(1, int(tracking_interval))
        self.general_logger = general_logger
        self.weight_logger = weight_logger
        self.input_output_logger = input_output_logger
        self.gradient_logger = gradient_logger
        self.writer = writer
        self.device_cpu_offload = bool(device_cpu_offload)
        self.enabled = enabled 

        # internals
        self._hooked = False

        self._tracked_modules: Dict[str, torch.nn.Module] = {}
        self._tracked_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        
        # We'll add one model-level pre-forward hook to increment the batch counter
        # once per forward (so monitoring checks run only once per batch)
        self._batch_counter = 0
        self._model_pre_forward_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    # -------------------------
    # Public API
    # -------------------------
    def hook(self):
        """Find modules by name and register forward/backward hooks."""
        if self._hooked:
            self.general_logger.warning("[HookHandler]: Already hooked, but the hook method is called again.")
            return

        # Map requested names -> actual modules (exact match on named_modules())
        for name, module in self.model.named_modules():
            if name in self.tracked_module_names:
                self._tracked_modules[name] = module

        # warn if any names not found
        missing = self.tracked_module_names - set(self._tracked_modules.keys())
        if missing:
            self.general_logger.warning(f"[HookHandler]: module(s) not found: {sorted(list(missing))}")

        # find all learnable weights in a module
        for mname, module in self._tracked_modules.items():
            tensor_dict = {}
            for pname, param in module.named_parameters(recurse=True):
                if param.requires_grad:
                    tensor_dict[pname] = param
            self._tracked_tensors[mname] = tensor_dict
        
        # register hooks for each tracked module
        for name, module in self._tracked_modules.items():
            h_fwd = module.register_forward_hook(self._make_forward_hook(name))
            h_bwd = None
            # register_full_backward_hook is recommended (PyTorch 1.8+)
            try:
                h_bwd = module.register_full_backward_hook(self._make_backward_hook(name))
            except Exception:
                assert False, "Backward hook registration failed. Try fixing it."
                # no fallback to older register_backward_hook because it is deprecated and buggy
                try:
                    h_bwd = module.register_backward_hook(self._make_backward_hook(name))
                except Exception:
                    h_bwd = None

            if h_fwd:
                self._handles.append(h_fwd)
            if h_bwd:
                self._handles.append(h_bwd)

        # model-level forward_pre_hook to increment batch counter once per forward
        self._model_pre_forward_handle = self.model.register_forward_pre_hook(self._batch_increment_hook)
        self._hooked = True
        self.general_logger.info(f"[HookHandler] Registered hooks for modules: {list(self._tracked_modules.keys())}")

    def unhook(self):
        """Remove all hooks."""
        for h in list(self._handles):
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        if self._model_pre_forward_handle is not None:
            try:
                self._model_pre_forward_handle.remove()
            except Exception:
                pass
            self._model_pre_forward_handle = None
        self._hooked = False
        self.general_logger.info("[HookHandler] All hooks removed.")

    def set_enabled(self, flag: bool):
        """Enable or disable hook tracking at runtime."""
        self.enabled = bool(flag)
    
    # Maybe this method can be used as a synchronized timer for some other hooks outside
    # so it is made public
    def should_monitor(self) -> bool:
        """Return True if monitoring should run this batch."""
        return (
            self.enabled and ((self._batch_counter % self.tracking_interval == 0) or self._batch_counter == 1)
        )
    
    # -------------------------
    # Hooks and hook creators
    # -------------------------
    def _make_forward_hook(self, module_name: str) -> Callable:

        def forward_hook(module, inputs, outputs):
            # Check if we should run monitoring for this batch
            if not self.should_monitor():
                return
            
            step = self._batch_counter

            with torch.no_grad():
                # 1) weights
                for wname, weight in self._tracked_tensors[module_name].items():
                    results = self._analyze_parameter(wname, weight)
                    self._log_tensor_stats(
                        stats=results, name=f"{module_name}/{wname}",
                        step=step, logger=self.weight_logger
                    )

                # 2) inputs and outputs
                # handle common cases on lists of inputs and outputs
                input_tensors = self._unpack_tensors(inputs)
                output_tensors = self._unpack_tensors(outputs)
                # if multiple output tensors, give them indices
                for i, t in enumerate(input_tensors):
                    tag = f"{module_name}/input" + (f"/{i}" if len(input_tensors) > 1 else "")
                    results = self._analyze_input_output(t)
                    self._log_tensor_stats(
                        stats=results, name=tag,
                        step=step, logger=self.input_output_logger
                    )

                for i, t in enumerate(output_tensors):
                    tag = f"{module_name}/output" + (f"/{i}" if len(output_tensors) > 1 else "")
                    results = self._analyze_input_output(t)
                    self._log_tensor_stats(
                        stats=results, name=tag,
                        step=step, logger=self.input_output_logger
                    )
        
        return forward_hook

    def _make_backward_hook(self, module_name: str) -> Callable:
        # backward hook signature differs slightly across APIs; both pass grad_input, grad_output tuples
        def backward_hook(module: torch.nn.Module, grad_input: Tuple, grad_output: Tuple):
            if not self.should_monitor():
                return
            for wname, weight in self._tracked_tensors[module_name].items():
                results = self._analyze_gradient(wname, weight)
                self._log_tensor_stats(
                    name=f"{module_name}/{wname}/gradient",
                    step=self._batch_counter, stats=results, 
                    logger=self.gradient_logger
                )
        return backward_hook

    def _batch_increment_hook(self, module, inputs):
        """Model-level pre-forward hook: increment batch counter once per forward call."""
        if module.training:
            self._batch_counter += 1
        return None
    
    # -------------------------
    # Helpers
    # -------------------------
    def _unpack_tensors(self, maybe_tuple):
        """Return a list of tensors found in maybe_tuple; if maybe_tuple is a single tensor, return [tensor]."""
        if isinstance(maybe_tuple, torch.Tensor):
            return [maybe_tuple]
        if maybe_tuple is None:
            return []
        if isinstance(maybe_tuple, (tuple, list)):
            tensors = []
            for x in maybe_tuple:
                if isinstance(x, torch.Tensor):
                    tensors.append(x)
                else:
                    # try to find tensors inside nested structures
                    if hasattr(x, '__iter__'):
                        for y in x:
                            if isinstance(y, torch.Tensor):
                                tensors.append(y)
                    else:
                        assert False, f"Too complex of a type for unpacking in hooks: {type(x)}"
            return tensors
        return []

    def _detach_and_to_cpu_and_to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """Detach and move to CPU and convert to numpy."""
        if not isinstance(t, torch.Tensor):
            raise TypeError("expected torch.Tensor")
        if self.device_cpu_offload:
            t = t.detach().cpu()
        else:
            t = t.detach()
        try:
            arr = t.numpy()
        except Exception:
            # fallback via .to('cpu') then numpy
            arr = t.cpu().numpy()
        return arr

    def _log_tensor_stats(
        self,
        name: str,
        step: int,
        stats: Dict[str, float],
        logger: Any
    ):
        # This may contain an error, which is intended to be written as if normally into the respective log file
        # In addition, the error has been written to the general log file
        logger.info(f"[{name} at step {step}] {stats}")

        POSSIBLE_SCALARS = [
            'coefficient_of_variation', 
            'mean', 'std', 'sparsity', 'l2_norm', 
            'spectral_norm', 'stable_rank', 
            'relative_norm', 'cosine_similarity',
            'cond', 'effective_rank',
            'top1_energy_ratio', 'top5_energy_ratio', 'top10_energy_ratio'
        ]
        # --- TensorBoard scalars ---
        if self.writer is not None:
            try:
                for k, v in stats.items():
                    if k in POSSIBLE_SCALARS:
                        self.writer.add_scalar(f"{name}/{k}", v, step)
            except Exception as e:
                try:
                    self.general_logger.error(f"[{name}] scalar write failed: {e}")
                except Exception:
                    pass
        
        POSSIBLE_HISTOGRAMS = ['singular_values']
        # --- TensorBoard histograms ---
        if self.writer is not None:
            try:
                for k, v in stats.items():
                    if k in POSSIBLE_HISTOGRAMS:
                        self.writer.add_histogram(f"{name}/{k}", v, step)
            except Exception as e:  
                try:
                    self.general_logger.error(f"[{name}] histogram write failed: {e}")
                except Exception:
                    pass
        
    def _analyze_parameter(self, pname, param: torch.Tensor):
        """
        Analyze a parameter tensor from a PyTorch model.

        - If it's a 1D bias → compute coefficient of variation & L2 norm.
        - If it's a 2D matrix → compute spectral norm, stable rank & up to top 100 sigular values.
        - If it's a convolutional kernel → reshape to (out_channels, in_channels*kernel_h*kernel_w),
          then treat as a 2D matrix.

        Returns a dictionary of results.
        """
        shape = param.shape
        arr = self._detach_and_to_cpu_and_to_numpy(param)
        results = {"shape": shape}

        # Case 1: Bias (1D tensor)
        if len(shape) == 1:
            results.update({"type": "vector or bias"})
            results.update(self._calculate_vector_metrics(arr))

        # Case 2: Linear / Fully Connected weight (2D tensor)
        elif len(shape) == 2:
            results.update({"type": "matrix"})
            results.update(self._calculate_matrix_spectrum(arr))

        # Case 3: Convolutional kernel (3D, 4D, 5D or 6D tensors are treated as convolutional kernels)
        elif len(shape) >= 3 and len(shape) <= 6:
            results.update({"type": "conv_kernel"})
            results.update(self._calculate_matrix_spectrum(arr))

        else:
            self.general_logger.error(f"Unsupported tensor {pname} of shape {shape}")
            results.update({
                "type": "unsupported",
                "message": f"Unsupported tensor {pname} of shape {shape}"
            })

        return results

    def _calculate_vector_metrics(self, v: np.ndarray, sparsity_eps: float = 5e-3):
        mean = np.mean(v)
        std = np.std(v)
        sparsity = float(np.mean(np.abs(v) <= sparsity_eps)) if v.size != 0 else 0.0
        l2_norm = np.linalg.norm(v, ord=2)
        return {
            "mean": mean,
            "std": std,
            "sparsity": sparsity,
            "l2_norm": l2_norm
        }

    def _calculate_matrix_spectrum(self, W: np.ndarray, max_rank: int = 1000000):
        """
        calculate singular value spectrum and stable rank of a given weight matrix

        Args:
            W (np.ndarray): Weight matrix, shape (m, n).
            writer (SummaryWriter): TensorBoard writer.
            tag (str): Name/identifier for the matrix (e.g. "layer1.fc").
            step (int): Current training step.
            max_rank (int): Max number of singular values to log.
        """
        # Ensure 2D (flatten conv filters if needed before calling this function)
        W = W.reshape(W.shape[0], -1)

        # Compute singular values
        sv = np.linalg.svd(W, compute_uv=False)
        sv = np.sort(sv)[::-1]  # descending order
        cond = sv.max() / max(sv.min(), np.finfo(float).eps)
        if len(sv) > max_rank:
            sv = sv[:max_rank]

        # Stable rank
        fro_norm_sq = np.sum(sv**2)
        spectral_norm_sq = sv[0]**2 if sv.size > 0 else 1e-12
        stable_rank = fro_norm_sq / spectral_norm_sq

        # --- Effective rank ---
        if np.sum(sv) > 0:
            p = sv / np.sum(sv)
            entropy = -np.sum(p * np.log(p + 1e-12))
            effective_rank = np.exp(entropy)
        else:
            effective_rank = 0.0

        # --- Top-k energy ratios ---
        energy_cumsum = np.cumsum(sv**2)
        total_energy = energy_cumsum[-1] if energy_cumsum.size > 0 else 1e-12
        def energy_ratio(k):
            return energy_cumsum[min(k, len(sv)) - 1] / total_energy

        topk_ratios = {
            'top1_energy_ratio': energy_ratio(1),
            'top5_energy_ratio': energy_ratio(5),
            'top10_energy_ratio': energy_ratio(10)
        }

        return {
            'singular_values': sv,
            'spectral_norm': sv[0],
            'stable_rank': stable_rank,
            'l2_norm': np.linalg.norm(W.ravel(), ord=2),
            'cond': cond,
            'effective_rank': effective_rank,
            **topk_ratios
        }

    def _analyze_input_output(self, input_or_output: torch.Tensor):
        """
        Given any PyTorch tensor, detach and flatten it,
        then compute some metrics on the vector.

        This tensor is supposed to be input or output tensor of a PyTorch module.

        Returns a dictionary with results.
        """
        arr = self._detach_and_to_cpu_and_to_numpy(input_or_output).ravel()
        results = {"shape": input_or_output.shape}
        results.update(self._calculate_vector_metrics(arr))
        return results

    def _analyze_gradient(self, pname, param: torch.Tensor):
        if param.grad is None:
            self.general_logger.error(f"error: No gradient found for parameter {pname}.")
            return {"error": f"No gradient found for parameter {pname}."}

        results = {"shape": param.grad.shape}
        p = self._detach_and_to_cpu_and_to_numpy(param).ravel()
        g = self._detach_and_to_cpu_and_to_numpy(param.grad).ravel()
        assert p.size == g.size, 'gradient analysis size exception'
        # assuming machine eps = 1e-7, target lr = 1e-4
        # g_threshold = machine_eps * ||p||_2 / target_lr
        # sparsity_eps = g_threshold / sqrt(N)
        sparsity_eps = (1e-7 * np.linalg.norm(p, ord=2)) / (1e-4 * np.sqrt(g.size)) 
        results.update(self._calculate_vector_metrics(g, sparsity_eps=sparsity_eps))
        return results


def build_hook_handler(
    args: argparse.Namespace,
    general_logger: logging.Logger,
    writer: Any,
    model: torch.nn.Module,
    tracked_module_names: Iterable[str],
    tracking_interval: int
):
    from util.logger import build_logger
    global general_hook_handler_built
    assert args, "An args object of the whole project from outside must be set for the hook handler"
    assert general_logger, "A general global logger of the whole project from outside must be set for the hook handler"
    if general_hook_handler_built:
        general_logger.info("[HookHandlerBuilder] Warning: hook handler already built once; skipping")
    general_hook_handler_built = True

    log_dir = args.save_path + "/model_monitoring"
    weight_logger = build_logger("weight_logger", log_dir, "weight_log")
    input_output_logger = build_logger("input_output_logger", log_dir, "input_output_log")
    gradient_logger = build_logger("gradient_logger", log_dir, "gradient_log")
    
    return GeneralHookHandler(
        model=model,
        tracked_module_names=tracked_module_names,
        tracking_interval=tracking_interval,
        general_logger=general_logger,
        weight_logger=weight_logger,
        input_output_logger=input_output_logger,
        gradient_logger=gradient_logger,
        writer=writer,
        device_cpu_offload=True,
        enabled=True
    )
