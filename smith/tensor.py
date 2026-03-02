"""
NanoTensor - Minimalist Automatic Differentiation Engine

Implements a reverse-mode AD engine using pure Python with
algebraic operations instead of if/else statements.
Deterministic Hardening: Topological sorting and Kahan summation for high precision.
"""

from typing import Tuple, Callable, List, Optional


class NanoTensor:
    """
    Lightweight tensor with automatic differentiation.
    Uses branchless algebraic primitives for all conditional logic.
    """
    
    def __init__(self, data, _parents=(), _op=None, requires_grad=True):
        if isinstance(data, (int, float)):
            data = [data]
        self.data = [float(x) for x in data] if isinstance(data, list) else data
        self.shape = (len(self.data),) if isinstance(self.data, list) else self.data.shape
        self.grad = [0.0] * len(self.data) if isinstance(self.data, list) else None
        # Gradient error for Kahan summation
        self._grad_err = [0.0] * len(self.data) if isinstance(self.data, list) else None

        self._backward = lambda: None
        self._parents = _parents
        self._op = _op
        self.requires_grad = requires_grad
        # Unique ID for deterministic sorting
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
        """Branchless max using algebraic formulation"""
        return (a + b + abs(a - b)) / 2.0
    
    @staticmethod
    def _min(a: float, b: float) -> float:
        """Branchless min using algebraic formulation"""
        return (a + b - abs(a - b)) / 2.0
    
    @staticmethod
    def _relu(x: float) -> float:
        """Branchless ReLU using max primitive"""
        return NanoTensor._max(x, 0.0)
    
    @staticmethod
    def _if_else(condition: float, true_val: float, false_val: float) -> float:
        """Branchless conditional using sign primitive"""
        mask = (NanoTensor._sign(condition) + 1.0) / 2.0
        return mask * true_val + (1.0 - mask) * false_val

    @staticmethod
    def _tanh(x: float) -> float:
        """Branchless tanh approximation"""
        x2 = x * x
        return x * (27.0 + x2) / (27.0 + 9.0 * x2 + 1e-7)

    @staticmethod
    def _exp(x: float) -> float:
        """Branchless exp approximation"""
        return 1.0 + x + 0.5 * x**2 + 0.1666 * x**3 + 0.0416 * x**4

    @staticmethod
    def _gelu(x: float) -> float:
        """Branchless GELU approximation"""
        return 0.5 * x * (1.0 + NanoTensor._tanh(0.79788456 * (x + 0.044715 * x**3)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Branchless sigmoid approximation"""
        return 0.5 * (x / (1.0 + abs(x)) + 1.0)
    
    def _accumulate_grad(self, index: int, value: float):
        """Accumulate gradient using Kahan Summation for precision and determinism."""
        if self.grad is None or not self.requires_grad:
            return

        # Kahan summation step
        y = value - self._grad_err[index]
        t = self.grad[index] + y
        self._grad_err[index] = (t - self.grad[index]) - y
        self.grad[index] = t

    # --- Tensor Operations ---
    def __add__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        out = NanoTensor([self.data[i] + other.data[i] for i in range(len(self.data))],
                         _parents=(self, other), _op='+')
        
        def _backward():
            for i in range(len(self.data)):
                self._accumulate_grad(i, out.grad[i])
                other._accumulate_grad(i, out.grad[i])
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        out = NanoTensor([self.data[i] * other.data[i] for i in range(len(self.data))],
                         _parents=(self, other), _op='*')
        
        def _backward():
            for i in range(len(self.data)):
                self._accumulate_grad(i, other.data[i] * out.grad[i])
                other._accumulate_grad(i, self.data[i] * out.grad[i])
        out._backward = _backward
        return out
    
    def matmul(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        m = len(self.data)
        n = len(other.data)

        if m == n:
            result = sum(self.data[i] * other.data[i] for i in range(n))
            out = NanoTensor([result], _parents=(self, other), _op='matmul_dot')

            def _backward():
                for i in range(len(self.data)):
                    self._accumulate_grad(i, other.data[i] * out.grad[0])
                    other._accumulate_grad(i, self.data[i] * out.grad[0])
            out._backward = _backward
            return out

        if n > 0 and m % n == 0:
            rows = m // n
            y = []
            for i in range(rows):
                base = i * n
                acc = 0.0
                for j in range(n):
                    acc += self.data[base + j] * other.data[j]
                y.append(acc)

            out = NanoTensor(y, _parents=(self, other), _op='matmul_mv')

            def _backward():
                for i in range(rows):
                    base = i * n
                    gi = out.grad[i]
                    for j in range(n):
                        self._accumulate_grad(base + j, other.data[j] * gi)

                for j in range(n):
                    acc = 0.0
                    for i in range(rows):
                        acc += self.data[i * n + j] * out.grad[i]
                    other._accumulate_grad(j, acc)

            out._backward = _backward
            return out

        raise AssertionError(f"Dimension mismatch in matmul: {m} and {n}")
    
    def relu(self):
        out = NanoTensor([self._relu(x) for x in self.data],
                         _parents=(self,), _op='relu')
        
        def _backward():
            for i in range(len(self.grad)):
                mask = (self._sign(self.data[i]) + 1.0) / 2.0
                self._accumulate_grad(i, mask * out.grad[i])
        out._backward = _backward
        return out

    def gelu(self):
        out = NanoTensor([self._gelu(x) for x in self.data],
                         _parents=(self,), _op='gelu')

        def _backward():
            for i in range(len(self.grad)):
                x = self.data[i]
                c = 0.79788456
                k = 0.044715
                inner = c * (x + k * x**3)
                tanh_inner = self._tanh(inner)
                sech_inner_sq = 1.0 - tanh_inner**2
                grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_inner_sq * c * (1.0 + 3.0 * k * x**2)
                self._accumulate_grad(i, grad * out.grad[i])
        out._backward = _backward
        return out

    def sigmoid(self):
        out = NanoTensor([self._sigmoid(x) for x in self.data],
                         _parents=(self,), _op='sigmoid')

        def _backward():
            for i in range(len(self.grad)):
                s = out.data[i]
                self._accumulate_grad(i, s * (1 - s) * out.grad[i])
        out._backward = _backward
        return out
    
    def sum(self):
        total = sum(self.data)
        out = NanoTensor([total], _parents=(self,), _op='sum')
        
        def _backward():
            for i in range(len(self.grad)):
                self._accumulate_grad(i, out.grad[0])
        out._backward = _backward
        return out
    
    def backward(self):
        """Deterministic backward pass using topological sort based on creation order."""
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                # Sort parents by creation index for strict determinism
                sorted_parents = sorted(t._parents, key=lambda x: x._creation_index)
                for parent in sorted_parents:
                    build_topo(parent)
                topo.append(t)

        build_topo(self)
        
        # Reset all gradients in the graph to zero initially
        # except the leaf node which we set to 1.0
        for t in topo:
            if t.grad:
                t.grad = [0.0] * len(t.grad)
                t._grad_err = [0.0] * len(t._grad_err)

        self.grad[0] = 1.0
        
        # Process in reverse topological order (linear pass)
        for t in reversed(topo):
            t._backward()
    
    def zero_grad(self):
        if self.grad:
            self.grad = [0.0] * len(self.grad)
            self._grad_err = [0.0] * len(self._grad_err)
    
    def __repr__(self):
        data_repr = self.data[:5] if len(self.data) > 5 else self.data
        grad_repr = [f'{g:.3f}' for g in self.grad[:5]] if self.grad and len(self.grad) > 5 else [f'{g:.3f}' for g in (self.grad or [])]
        return f"NanoTensor(data={data_repr}{'...' if len(self.data) > 5 else ''}, grad={grad_repr}{'...' if self.grad and len(self.grad) > 5 else ''})"
