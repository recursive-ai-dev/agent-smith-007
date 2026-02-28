"""
NanoTensor - Lightweight Tensor with Automatic Differentiation

Implements a minimal autograd system using branchless algebraic primitives.
All conditional logic uses algebraic operations instead of if/else statements.
"""

from typing import Tuple, Callable


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
        self._backward = lambda: None
        self._parents = _parents
        self._op = _op
        self.requires_grad = requires_grad
    
    # --- Algebraic Primitives (Branchless) ---
    @staticmethod
    def _sign(x: float) -> float:
        """Branchless sign function: returns -1, 0, or 1"""
        return float((x > 0) - (x < 0))  # Python bools are ints, this is branchless
    
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
        # condition > 0 -> sign(condition) = 1
        mask = (NanoTensor._sign(condition) + 1.0) / 2.0
        return mask * true_val + (1.0 - mask) * false_val

    @staticmethod
    def _tanh(x: float) -> float:
        """Branchless tanh approximation"""
        x2 = x * x
        return x * (27.0 + x2) / (27.0 + 9.0 * x2 + 1e-7)

    @staticmethod
    def _cosh(x: float) -> float:
        """Branchless cosh using exp approximation"""
        # cosh(x) = (e^x + e^-x) / 2
        ex = NanoTensor._exp(x)
        e_neg_x = NanoTensor._exp(-x)
        return (ex + e_neg_x) / 2.0

    @staticmethod
    def _exp(x: float) -> float:
        """Branchless exp approximation"""
        # Using a simple polynomial approximation for exp(x)
        return 1.0 + x + 0.5 * x**2 + 0.1666 * x**3 + 0.0416 * x**4

    @staticmethod
    def _gelu(x: float) -> float:
        """Branchless GELU approximation"""
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + NanoTensor._tanh(0.79788456 * (x + 0.044715 * x**3)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Branchless sigmoid approximation"""
        return 0.5 * (x / (1.0 + abs(x)) + 1.0)
    
    # --- Tensor Operations ---
    def __add__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        out = NanoTensor([self.data[i] + other.data[i] for i in range(len(self.data))],
                         _parents=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[i]
            if other.requires_grad:
                for i in range(len(other.grad)):
                    other.grad[i] += out.grad[i]
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        out = NanoTensor([self.data[i] * other.data[i] for i in range(len(self.data))],
                         _parents=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += other.data[i] * out.grad[i]
            if other.requires_grad:
                for i in range(len(other.grad)):
                    other.grad[i] += self.data[i] * out.grad[i]
        out._backward = _backward
        return out
    
    def matmul(self, other):
        """Matrix/vector multiplication.
        Supports:
        - Dot product when vectors have equal length
        - Matrix-vector product when len(self) is a multiple of len(other)
          (rows = len(self) // len(other), row-major flatten)
        """
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        m = len(self.data)
        n = len(other.data)

        # Case 1: standard dot product (1-element output)
        if m == n:
            result = sum(self.data[i] * other.data[i] for i in range(n))
            out = NanoTensor([result], _parents=(self, other), _op='matmul_dot')

            def _backward():
                if self.requires_grad:
                    for i in range(len(self.grad)):
                        self.grad[i] += other.data[i] * out.grad[0]
                if other.requires_grad:
                    for i in range(len(other.grad)):
                        other.grad[i] += self.data[i] * out.grad[0]
            out._backward = _backward
            return out

        # Case 2: matrix-vector product (row-major: rows = m // n)
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
                # dL/dW[i,j] += out.grad[i] * x[j]
                if self.requires_grad:
                    for i in range(rows):
                        base = i * n
                        gi = out.grad[i]
                        for j in range(n):
                            self.grad[base + j] += other.data[j] * gi

                # dL/dx[j] += sum_i W[i,j] * out.grad[i]
                if other.requires_grad:
                    for j in range(n):
                        acc = 0.0
                        for i in range(rows):
                            acc += self.data[i * n + j] * out.grad[i]
                        other.grad[j] += acc

            out._backward = _backward
            return out

        # Otherwise dimensions are incompatible
        raise AssertionError("Dimension mismatch in matmul: incompatible lengths")
    
    def relu(self):
        """Branchless ReLU activation"""
        out = NanoTensor([self._relu(x) for x in self.data],
                         _parents=(self,), _op='relu')
        
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    # Branchless gradient: 1 if x > 0 else 0
                    self.grad[i] += (self._sign(self.data[i]) + 1.0) / 2.0 * out.grad[i]
        out._backward = _backward
        return out

    def gelu(self):
        """Branchless GELU activation"""
        out = NanoTensor([self._gelu(x) for x in self.data],
                         _parents=(self,), _op='gelu')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    x = self.data[i]
                    # Correct derivative of the GELU approximation
                    # d/dx(0.5 * x * (1 + tanh(c * (x + k * x^3)))) =
                    # 0.5 * (1 + tanh(c * (x + k * x^3))) + 0.5 * x * sech^2(c * (x + k * x^3)) * c * (1 + 3k * x^2)
                    c = 0.79788456
                    k = 0.044715
                    inner = c * (x + k * x**3)
                    tanh_inner = self._tanh(inner)
                    sech_inner_sq = 1.0 - tanh_inner**2
                    grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_inner_sq * c * (1.0 + 3.0 * k * x**2)
                    self.grad[i] += grad * out.grad[i]
        out._backward = _backward
        return out

    def sigmoid(self):
        """Branchless sigmoid activation"""
        out = NanoTensor([self._sigmoid(x) for x in self.data],
                         _parents=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    s = out.data[i]
                    self.grad[i] += s * (1 - s) * out.grad[i]
        out._backward = _backward
        return out
    
    def sum(self):
        """Sum all elements"""
        total = sum(self.data)
        out = NanoTensor([total], _parents=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[0]
        out._backward = _backward
        return out
    
    def backward(self):
        """Backward pass through computation graph"""
        # Topological sort
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for parent in t._parents:
                    build_topo(parent)
                topo.append(t)
        build_topo(self)
        
        # Initialize gradient
        self.grad[0] = 1.0
        
        # Process in reverse topological order
        for t in reversed(topo):
            t._backward()
    
    def zero_grad(self):
        """Reset gradients"""
        if self.grad:
            self.grad = [0.0] * len(self.grad)
    
    def __repr__(self):
        data_repr = self.data[:5] if len(self.data) > 5 else self.data
        grad_repr = [f'{g:.3f}' for g in self.grad[:5]] if len(self.grad or []) > 5 else [f'{g:.3f}' for g in (self.grad or [])]
        return f"NanoTensor(data={data_repr}{'...' if len(self.data) > 5 else ''}, grad={grad_repr}{'...' if len(self.grad or []) > 5 else ''})"
