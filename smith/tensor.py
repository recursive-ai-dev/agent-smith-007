"""
NanoTensor - Lightweight Tensor with Automatic Differentiation
High-Fidelity Refactor: AtomicGradient Edition

Extended with full mathematical operations for AgentSmith classifier:
  neg, sub, div, pow, exp, log, sqrt, tanh, softmax, mean, concat,
  weighted_sum, extract — all with correct analytical backpropagation.
"""

import math
from typing import List, Tuple, Optional, Callable, Set, Union, Dict, Any


class NanoTensor:
    """
    Lightweight tensor with automatic differentiation and high-precision branchless primitives.
    Implements logical atomicity in the computation graph to prevent torn-writes.
    """
    
    def __init__(
        self,
        data: Union[float, List[float]],
        _parents: Tuple['NanoTensor', ...] = (),
        _op: Optional[str] = None,
        requires_grad: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if isinstance(data, (int, float)):
            data = [float(data)]
        self.data = [float(x) for x in data]
        self.shape = (len(self.data),)
        self.grad = [0.0] * len(self.data)
        # Gradient error for Kahan summation to maintain high precision during accumulation
        self._grad_err = [0.0] * len(self.data)

        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op
        self.requires_grad = requires_grad
        self.metadata = metadata or {}
        # Unique ID for deterministic sorting in topological sort
        self._creation_index = self._get_next_index()
    
    _global_index = 0
    @classmethod
    def _get_next_index(cls):
        cls._global_index += 1
        return cls._global_index

    # --- Algebraic Primitives (Branchless) ---
    @staticmethod
    def _sign(x: float) -> float:
        """Branchless sign function: returns -1, 0, or 1"""
        return float((x > 0) - (x < 0))
    
    @staticmethod
    def _max(a: float, b: float) -> float:
        """Branchless max function"""
        return (a + b + abs(a - b)) / 2.0
    
    @staticmethod
    def _min(a: float, b: float) -> float:
        """Branchless min function"""
        return (a + b - abs(a - b)) / 2.0

    @staticmethod
    def _relu(x: float) -> float:
        """Branchless ReLU"""
        return NanoTensor._max(x, 0.0)
    
    @staticmethod
    def _if_else(condition: float, true_val: float, false_val: float) -> float:
        """Branchless conditional using sign primitive"""
        mask = (NanoTensor._sign(condition) + 1.0) / 2.0
        return mask * true_val + (1.0 - mask) * false_val

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Stable sigmoid function for internal use in GELU/Tanh approximations"""
        try:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    @staticmethod
    def _exp(x: float) -> float:
        """Branchless exp approximation (Taylor expansion)"""
        return 1.0 + x + 0.5 * x**2 + 0.16666666666666666 * x**3 + 0.04166666666666666 * x**4

    @staticmethod
    def _tanh(x: float) -> float:
        """Standard math.tanh wrapper for primitives"""
        return math.tanh(x)

    @staticmethod
    def _gelu(x: float) -> float:
        """Branchless GELU approximation used in FFN layers"""
        return 0.5 * x * (1.0 + NanoTensor._tanh(0.79788456 * (x + 0.044715 * x**3)))
    
    def _accumulate_grad(self, index: int, value: float):
        """Accumulate gradient using Kahan Summation for precision and determinism."""
        if self.grad is None or not self.requires_grad:
            return

        # Kahan summation step: minimizes floating point errors during large-scale backprop
        y = value - self._grad_err[index]
        t = self.grad[index] + y
        self._grad_err[index] = (t - self.grad[index]) - y
        self.grad[index] = t

    def sigmoid(self) -> 'NanoTensor':
        """Element-wise sigmoid with backprop."""
        out_data = [self._sigmoid(x) for x in self.data]
        out = NanoTensor(out_data, _parents=(self,), _op='sigmoid')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    # derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
                    s = out.data[i]
                    self._accumulate_grad(i, s * (1.0 - s) * out.grad[i])
        out._backward = _backward
        return out

    # --- Tensor Operations ---

    def __add__(self, other: Union['NanoTensor', float, int]) -> 'NanoTensor':
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        s_data, o_data = self.data, other.data

        # Broadcasting logic (restricted to scalar or exact match)
        if len(s_data) != len(o_data):
            if len(o_data) == 1:
                o_data_eff = [o_data[0]] * len(s_data)
                s_data_eff = s_data
            elif len(s_data) == 1:
                s_data_eff = [s_data[0]] * len(o_data)
                o_data_eff = o_data
            else:
                raise ValueError(f"Dimension mismatch for addition: {len(s_data)} and {len(o_data)}")
        else:
            s_data_eff, o_data_eff = s_data, o_data

        out = NanoTensor([s1 + o1 for s1, o1 in zip(s_data_eff, o_data_eff)],
                         _parents=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                if len(self.data) == 1 and len(out.data) > 1:
                    # Broadcast backward
                    self._accumulate_grad(0, sum(out.grad))
                else:
                    for i in range(len(self.grad)):
                        self._accumulate_grad(i, out.grad[i])
            if other.requires_grad:
                if len(other.data) == 1 and len(out.data) > 1:
                    # Broadcast backward
                    other._accumulate_grad(0, sum(out.grad))
                else:
                    for i in range(len(other.grad)):
                        other._accumulate_grad(i, out.grad[i])
        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> 'NanoTensor':
        return self + other

    def __neg__(self) -> 'NanoTensor':
        return self * -1.0

    def __sub__(self, other: Union['NanoTensor', float, int]) -> 'NanoTensor':
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> 'NanoTensor':
        return NanoTensor(other) + (-self)

    def __mul__(self, other: Union['NanoTensor', float, int]) -> 'NanoTensor':
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        s_data, o_data = self.data, other.data

        # Broadcasting logic
        if len(s_data) != len(o_data):
            if len(o_data) == 1:
                o_data_eff = [o_data[0]] * len(s_data)
                s_data_eff = s_data
            elif len(s_data) == 1:
                s_data_eff = [s_data[0]] * len(o_data)
                o_data_eff = o_data
            else:
                raise ValueError(f"Dimension mismatch for multiplication: {len(s_data)} and {len(o_data)}")
        else:
            s_data_eff, o_data_eff = s_data, o_data

        out = NanoTensor([s1 * o1 for s1, o1 in zip(s_data_eff, o_data_eff)],
                         _parents=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                if len(self.data) == 1 and len(out.data) > 1:
                    self._accumulate_grad(0, sum(o_data_eff[i] * out.grad[i] for i in range(len(out.data))))
                else:
                    for i in range(len(self.grad)):
                        self._accumulate_grad(i, o_data_eff[i] * out.grad[i])
            if other.requires_grad:
                if len(other.data) == 1 and len(out.data) > 1:
                    other._accumulate_grad(0, sum(s_data_eff[i] * out.grad[i] for i in range(len(out.data))))
                else:
                    for i in range(len(other.grad)):
                        other._accumulate_grad(i, s_data_eff[i] * out.grad[i])
        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> 'NanoTensor':
        return self * other

    def matmul(self, other: 'NanoTensor') -> 'NanoTensor':
        """
        Matrix-vector or dot product multiplication.
        If self is a matrix (flat array), 'other' must be a vector.
        """
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        m = len(self.data)
        n = len(other.data)

        if m == n: # Dot product case
            res = sum(self.data[i] * other.data[i] for i in range(m))
            out = NanoTensor([res], _parents=(self, other), _op='dot')
            def _backward():
                if self.requires_grad:
                    for i in range(m):
                        self._accumulate_grad(i, other.data[i] * out.grad[0])
                if other.requires_grad:
                    for i in range(n):
                        other._accumulate_grad(i, self.data[i] * out.grad[0])
            out._backward = _backward
            return out

        if n > 0 and m % n == 0: # Matrix-Vector product case
            rows = m // n
            y = []
            for i in range(rows):
                y.append(sum(self.data[i*n + j] * other.data[j] for j in range(n)))
            out = NanoTensor(y, _parents=(self, other), _op='mv')
            def _backward():
                if self.requires_grad:
                    # ∂L/∂W_ij = ∂L/∂y_i * x_j
                    for i in range(rows):
                        for j in range(n):
                            self._accumulate_grad(i*n + j, other.data[j] * out.grad[i])
                if other.requires_grad:
                    # ∂L/∂x_j = Σ_i ∂L/∂y_i * W_ij
                    for j in range(n):
                        val = sum(out.grad[i] * self.data[i*n + j] for i in range(rows))
                        other._accumulate_grad(j, val)
            out._backward = _backward
            return out
        
        raise ValueError(f"Incompatible shapes for matmul: {m} and {n}")

    def __truediv__(self, other: Union['NanoTensor', float, int]) -> 'NanoTensor':
        if isinstance(other, (int, float)):
            return self * NanoTensor([1.0 / other], requires_grad=False)
        return self * other.reciprocal()

    def __rtruediv__(self, other: Union[float, int]) -> 'NanoTensor':
        return NanoTensor([float(other)], requires_grad=False) * self.reciprocal()

    def reciprocal(self) -> 'NanoTensor':
        """Element-wise 1/x.  ∂(1/x_i)/∂x_i = -1/x_i²."""
        # Use small epsilon to prevent absolute zero division, preserving sign
        eps = 1e-30
        safe = [x if abs(x) > eps else (eps if x >= 0 else -eps) for x in self.data]
        out = NanoTensor([1.0 / s for s in safe], _parents=(self,), _op='recip')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    # derivative of 1/x is -1/x^2
                    self._accumulate_grad(i, -out.data[i] ** 2 * out.grad[i])
        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> 'NanoTensor':
        """Element-wise power x^k.  ∂(x_i^k)/∂x_i = k * x_i^(k-1)."""
        out = NanoTensor([x ** exponent for x in self.data], _parents=(self,), _op=f'pow{exponent}')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    base = self.data[i]
                    if base == 0.0 and exponent < 1:
                        # Gradient is undefined at 0 for roots; set to 0 to prevent NaN
                        self._accumulate_grad(i, 0.0)
                    elif base == 0.0 and exponent == 1:
                        self._accumulate_grad(i, out.grad[i])
                    elif base == 0.0:
                        self._accumulate_grad(i, 0.0)
                    else:
                        self._accumulate_grad(i, exponent * (base ** (exponent - 1.0)) * out.grad[i])
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Transcendental operations
    # ------------------------------------------------------------------

    def exp(self) -> 'NanoTensor':
        """Element-wise e^x.  ∂(e^x_i)/∂x_i = e^x_i."""
        # Clamp to prevent infinity/overflow in training
        out = NanoTensor([math.exp(min(x, 88.72)) for x in self.data], _parents=(self,), _op='exp')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self._accumulate_grad(i, out.data[i] * out.grad[i])
        out._backward = _backward
        return out

    def log(self) -> 'NanoTensor':
        """Element-wise natural log.  ∂log(x_i)/∂x_i = 1/x_i."""
        eps = 1e-30
        clamped = [x < eps for x in self.data]
        safe = [max(x, eps) for x in self.data]
        out = NanoTensor([math.log(s) for s in safe], _parents=(self,), _op='log')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    if clamped[i]:
                        continue  # zero gradient for originally invalid inputs to maintain stability
                    self._accumulate_grad(i, (1.0 / safe[i]) * out.grad[i])
        out._backward = _backward
        return out

    def sqrt(self) -> 'NanoTensor':
        """Element-wise sqrt(x).  ∂sqrt(x_i)/∂x_i = 1 / (2*sqrt(x_i))."""
        eps = 1e-30
        clamped = [x < eps for x in self.data]
        safe = [max(x, eps) for x in self.data]
        out = NanoTensor([math.sqrt(s) for s in safe], _parents=(self,), _op='sqrt')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    if clamped[i]:
                        continue  # stability clamp
                    self._accumulate_grad(i, 0.5 / out.data[i] * out.grad[i])
        out._backward = _backward
        return out

    def tanh(self) -> 'NanoTensor':
        """Element-wise tanh.  ∂tanh(x_i)/∂x_i = 1 - tanh(x_i)²."""
        out = NanoTensor([math.tanh(x) for x in self.data], _parents=(self,), _op='tanh')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self._accumulate_grad(i, (1.0 - out.data[i] ** 2) * out.grad[i])
        out._backward = _backward
        return out

    def gelu(self) -> 'NanoTensor':
        """Element-wise GELU approximation. Uses exact backward based on analytical form."""
        out_data = [NanoTensor._gelu(x) for x in self.data]
        out = NanoTensor(out_data, _parents=(self,), _op='gelu')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    x = self.data[i]
                    # GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/pi)*(x + 0.044715x^3)))
                    # Using derivative of the approximation for backprop
                    inner = 0.79788456 * (x + 0.044715 * x**3)
                    t = math.tanh(inner)
                    dt = (1.0 - t**2) * 0.79788456 * (1.0 + 3.0 * 0.044715 * x**2)
                    dg = 0.5 * (1.0 + t) + 0.5 * x * dt
                    self._accumulate_grad(i, dg * out.grad[i])
        out._backward = _backward
        return out

    def mean(self) -> 'NanoTensor':
        """Mean over all elements → scalar.  ∂mean/∂x_i = 1/n."""
        n = len(self.data)
        if n == 0:
            raise ValueError("Cannot compute mean of empty tensor")
        out = NanoTensor([sum(self.data) / n], _parents=(self,), _op='mean')
        def _backward():
            if self.requires_grad:
                grad_val = out.grad[0] / n
                for i in range(n):
                    self._accumulate_grad(i, grad_val)
        out._backward = _backward
        return out

    def sum(self) -> 'NanoTensor':
        """Sum over all elements → scalar. ∂sum/∂x_i = 1."""
        out = NanoTensor([sum(self.data)], _parents=(self,), _op='sum')
        def _backward():
            if self.requires_grad:
                for i in range(len(self.data)):
                    self._accumulate_grad(i, out.grad[0])
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Numerically-stable softmax with analytical Jacobian backprop
    # ------------------------------------------------------------------

    def softmax(self) -> 'NanoTensor':
        """
        Softmax over all elements.
        Forward:  p_i = exp(x_i - max_x) / sum_j exp(x_j - max_x)
        Backward: ∂L/∂x_i = p_i (∂L/∂p_i  − Σ_j p_j ∂L/∂p_j)
        """
        if not self.data:
            raise ValueError("Softmax requires non-empty tensor")
        max_x = max(self.data)
        e = [math.exp(xi - max_x) for xi in self.data]
        s = sum(e)
        probs = [ei / s for ei in e]
        out = NanoTensor(probs[:], _parents=(self,), _op='softmax')
        def _backward():
            if self.requires_grad:
                # Dot product of probabilities and incoming gradients
                dot = sum(probs[i] * out.grad[i] for i in range(len(probs)))
                for i in range(len(self.grad)):
                    # Softmax local Jacobian: diag(p) - pp^T
                    self._accumulate_grad(i, probs[i] * (out.grad[i] - dot))
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Structural: concat, extract (slice), weighted_sum
    # ------------------------------------------------------------------

    def concat(self, other: 'NanoTensor') -> 'NanoTensor':
        """
        Concatenate two 1-D NanoTensors → [n1 + n2].
        Gradient fan-out: upstream grad routed back to each source region.
        """
        n1, n2 = len(self.data), len(other.data)
        out = NanoTensor(self.data + other.data, _parents=(self, other), _op='concat')
        def _backward():
            if self.requires_grad:
                for i in range(n1):
                    self._accumulate_grad(i, out.grad[i])
            if other.requires_grad:
                for i in range(n2):
                    other._accumulate_grad(i, out.grad[n1 + i])
        out._backward = _backward
        return out

    def extract(self, start: int, end: int) -> 'NanoTensor':
        """
        Extract a contiguous slice [start, end).
        ∂L/∂x_i = ∂L/∂out_{i-start}  for i in [start, end), else 0.
        """
        length = end - start
        if start < 0 or end > len(self.data) or start > end:
            raise IndexError(f"Invalid slice indices for extraction: [{start}:{end}] for length {len(self.data)}")
        out = NanoTensor(self.data[start:end], _parents=(self,), _op='extract')
        def _backward():
            if self.requires_grad:
                for i in range(length):
                    self._accumulate_grad(start + i, out.grad[i])
        out._backward = _backward
        return out

    @staticmethod
    def weighted_sum(weights: 'NanoTensor', values: 'List[NanoTensor]') -> 'NanoTensor':
        """
        Bilinear attention context:  out[d] = Σ_j weights[j] * values[j][d]

        weights : NanoTensor [seq_len]
        values  : list of seq_len NanoTensors, each [d_v]
        Returns : NanoTensor [d_v]
        """
        seq_len = len(weights.data)
        if seq_len == 0 or not values:
            raise ValueError("weighted_sum requires non-empty inputs")

        d_v = len(values[0].data)
        result = [
            sum(weights.data[j] * values[j].data[d] for j in range(seq_len))
            for d in range(d_v)
        ]
        out = NanoTensor(result, _parents=tuple([weights] + values), _op='wsum')
        def _backward():
            if weights.requires_grad:
                for j in range(seq_len):
                    # ∂L/∂weights[j] = Σ_d ∂L/∂out[d] * values[j][d]
                    gw = sum(out.grad[d] * values[j].data[d] for d in range(d_v))
                    weights._accumulate_grad(j, gw)
            for j, v in enumerate(values):
                if v.requires_grad:
                    for d in range(d_v):
                        # ∂L/∂values[j][d] = weights[j] * ∂L/∂out[d]
                        v._accumulate_grad(d, weights.data[j] * out.grad[d])
        out._backward = _backward
        return out

    def backward(self):
        """
        Iterative topological sort + reverse-mode AD.
        Uses explicit stack (no recursion) to handle arbitrarily deep computation graphs.
        Gradient accumulation via Kahan summation.
        """
        # Iterative post-order DFS to build topological ordering
        topo: List['NanoTensor'] = []
        visited: Set['NanoTensor'] = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            # Sort parents by creation index for deterministic graph traversal
            for parent in sorted(node._parents, key=lambda x: x._creation_index):
                if parent not in visited:
                    stack.append((parent, False))

        # Zero all gradients in the reachable graph before backprop
        for t in topo:
            if t.grad is not None:
                t.grad = [0.0] * len(t.grad)
                t._grad_err = [0.0] * len(t._grad_err)

        # Seed gradient at the root (must be a scalar for standard backward)
        if len(self.data) != 1:
            raise ValueError(
                f"backward() requires a scalar output; got shape {len(self.data)}"
            )
        self.grad[0] = 1.0

        # Propagate in reverse topological order
        for t in reversed(topo):
            t._backward()
    
    def zero_grad(self):
        """Reset gradients and error accumulators to zero."""
        self.grad = [0.0] * len(self.data)
        self._grad_err = [0.0] * len(self.data)
    
    def __repr__(self) -> str:
        data_repr = self.data[:5] if len(self.data) > 5 else self.data
        grad_repr = [f'{g:.3f}' for g in self.grad[:5]] if self.grad and len(self.grad) > 5 else [f'{g:.3f}' for g in (self.grad or [])]
        return f"NanoTensor(data={data_repr}{'...' if len(self.data) > 5 else ''}, grad={grad_repr}{'...' if self.grad and len(self.grad) > 5 else ''})"
