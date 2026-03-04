"""
Data Pipeline
=============
Provides:
  1. A built-in synthetic dataset with representative text snippets for
     all 12 classification domains.  This lets the model train immediately
     without any external files.
  2. A CSV / plain-text file loader for real datasets.
  3. A DataLoader that shuffles and iterates batches of (text, label) pairs.

Data format expected by the model
-----------------------------------
  Each sample: (text: str, label: int)
  label ∈ {0, 1, …, 11}  (matching AgentSmithConfig.domains ordering)
"""

import csv
import math
import os
import random
from typing import List, Optional, Tuple

from ..classifier.config import DOMAINS

# ─────────────────────────────────────────────────────────────────────────────
# Built-in synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DATA: List[Tuple[str, int]] = [
    # 0 — science_technology
    ("The Large Hadron Collider accelerates protons to nearly the speed of light.",              0),
    ("Quantum entanglement allows particles to be correlated regardless of distance.",           0),
    ("CRISPR gene-editing technology allows precise modification of DNA sequences.",             0),
    ("Solar panels convert photons into electrical current via the photoelectric effect.",       0),
    ("The semiconductor transistor is the fundamental building block of modern computing.",      0),
    ("Nanotechnology manipulates matter at the atomic and molecular scale.",                     0),
    ("Fusion reactors aim to replicate the energy-producing process of the sun.",                0),
    ("Dark matter constitutes approximately 27 percent of the universe's total mass-energy.",    0),
    ("5G networks use millimeter-wave frequencies for ultra-high-speed wireless communication.", 0),
    ("Quantum computing leverages superposition and entanglement to process information.",       0),

    # 1 — mathematics
    ("The Riemann hypothesis concerns the distribution of prime numbers on the complex plane.",  1),
    ("Euler's identity relates five fundamental mathematical constants in one equation.",         1),
    ("The Pythagorean theorem states that a squared plus b squared equals c squared.",           1),
    ("Topology studies properties preserved under continuous deformations.",                     1),
    ("The Fourier transform decomposes a function into its frequency components.",               1),
    ("A prime number has exactly two distinct positive divisors: one and itself.",               1),
    ("Calculus provides tools for computing derivatives and integrals of functions.",            1),
    ("Linear algebra deals with vector spaces and linear mappings between them.",                1),
    ("The Mandelbrot set is defined by a recursive formula in the complex number plane.",        1),
    ("Group theory studies algebraic structures known as groups and their symmetries.",          1),

    # 2 — medicine_health
    ("Antibiotics target bacterial cell walls and disrupt protein synthesis pathways.",          2),
    ("The immune system uses T-cells and B-cells to fight pathogens and disease.",               2),
    ("Hypertension increases the risk of stroke, heart attack, and kidney disease.",             2),
    ("mRNA vaccines instruct cells to produce spike proteins that train the immune system.",     2),
    ("Oncology focuses on the diagnosis and treatment of cancerous tumors.",                     2),
    ("Insulin regulates blood glucose by facilitating cellular uptake of sugar.",                2),
    ("Anesthesia renders patients unconscious and insensible to pain during surgery.",           2),
    ("Epidemiology studies the distribution and determinants of health-related states.",         2),
    ("Neuroplasticity refers to the brain's ability to reorganise its synaptic connections.",   2),
    ("Stem cells can self-renew and differentiate into specialised cell types.",                 2),

    # 3 — law_legal
    ("Habeas corpus is a legal action requiring a detained person to be brought before a court.",3),
    ("Tort law governs civil wrongs that cause harm or loss to another person.",                 3),
    ("The due process clause protects citizens from arbitrary deprivation of rights.",           3),
    ("Intellectual property law covers patents, trademarks, and copyrights.",                    3),
    ("Contract law requires offer, acceptance, and consideration for a binding agreement.",      3),
    ("Criminal law defines offences against the state and specifies corresponding penalties.",   3),
    ("Judicial review allows courts to invalidate legislation that violates the constitution.",  3),
    ("Stare decisis binds lower courts to follow precedents set by higher courts.",              3),
    ("Liability in negligence requires duty of care, breach, causation, and damages.",          3),
    ("The rule of law mandates that all persons are accountable to laws equally enforced.",      3),

    # 4 — finance_economics
    ("The efficient market hypothesis holds that asset prices reflect all available information.",4),
    ("Compound interest grows exponentially as interest accrues on accumulated interest.",       4),
    ("Central banks use monetary policy to control inflation and stabilise the economy.",        4),
    ("Supply and demand curves intersect at the equilibrium price and quantity.",                4),
    ("A bond is a fixed-income instrument representing a loan made by an investor.",             4),
    ("Gross domestic product measures the total value of goods and services produced.",          4),
    ("Hedge funds employ leverage and derivatives to achieve absolute returns.",                  4),
    ("The Black-Scholes model prices European options using stochastic differential equations.", 4),
    ("Fiscal policy involves government spending and taxation to influence the economy.",         4),
    ("Portfolio diversification reduces unsystematic risk by holding uncorrelated assets.",      4),

    # 5 — literature_arts
    ("Hamlet's soliloquy explores themes of mortality, indecision, and the nature of being.",   5),
    ("Modernism in literature broke from tradition by employing stream-of-consciousness prose.", 5),
    ("The Impressionist painters captured fleeting light effects through loose brushwork.",      5),
    ("Sonnets follow a fourteen-line structure with a specific rhyme scheme.",                   5),
    ("Magical realism blends realistic narrative with fantastical or mythical elements.",        5),
    ("Jazz improvisation draws on blues scales, syncopation, and call-and-response patterns.",   5),
    ("Metaphor substitutes one thing for another to highlight a particular quality.",            5),
    ("Renaissance art revived classical ideals of proportion, harmony, and human dignity.",      5),
    ("Surrealism sought to release the creative potential of the subconscious mind.",            5),
    ("Narrative tension arises from conflict between characters, ideas, or inner desires.",      5),

    # 6 — history_politics
    ("The French Revolution overthrew the monarchy and established principles of liberty.",      6),
    ("The Cold War was a geopolitical rivalry between the United States and Soviet Union.",      6),
    ("Democracy is a system of government in which power is vested in the people.",              6),
    ("The Treaty of Versailles formally ended World War One and imposed reparations on Germany.",6),
    ("Colonialism involved the domination and exploitation of one nation by another.",           6),
    ("The Industrial Revolution transformed manufacturing through steam power and factories.",   6),
    ("Nationalism is an ideology centred on the idea that the nation is the core political unit.",6),
    ("The United Nations was established after World War Two to maintain international peace.",  6),
    ("Totalitarianism is a system in which a single party holds absolute political control.",    6),
    ("The civil rights movement fought for racial equality under the law in the United States.", 6),

    # 7 — philosophy_ethics
    ("Kant's categorical imperative demands that we act only according to universal maxims.",    7),
    ("Utilitarianism holds that the morally right action maximises overall happiness.",           7),
    ("Existentialism asserts that existence precedes essence and individuals create meaning.",    7),
    ("Epistemology investigates the nature, sources, and limits of human knowledge.",            7),
    ("Plato's allegory of the cave illustrates the distinction between illusion and reality.",   7),
    ("Virtue ethics focuses on character and the cultivation of moral virtues.",                 7),
    ("Determinism holds that all events are causally necessitated by prior states.",             7),
    ("The trolley problem is a thought experiment in moral philosophy about sacrifice.",         7),
    ("Phenomenology studies structures of conscious experience from the first-person view.",     7),
    ("Social contract theory posits that political authority derives from the consent of individuals.",7),

    # 8 — engineering
    ("A bridge must withstand tension, compression, and shear forces simultaneously.",           8),
    ("Thermodynamics governs energy conversion and the limits of heat engine efficiency.",       8),
    ("Signal processing uses filters to remove noise and extract useful information.",           8),
    ("Control systems use feedback to regulate the behaviour of dynamic systems.",               8),
    ("Fluid dynamics describes the motion of liquids and gases under various conditions.",       8),
    ("Material fatigue occurs when cyclic stresses cause microscopic cracks to propagate.",      8),
    ("PID controllers compute a correction based on proportional, integral, and derivative terms.",8),
    ("Structural analysis determines the internal forces and deformations in a loaded structure.",8),
    ("Electrical impedance is the complex generalisation of resistance to alternating current.", 8),
    ("Reinforced concrete combines the compressive strength of concrete with steel's tensile strength.",8),

    # 9 — natural_sciences
    ("Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",     9),
    ("Plate tectonics explains the movement of lithospheric plates and seismic activity.",       9),
    ("DNA replication ensures genetic information is faithfully copied before cell division.",   9),
    ("Osmosis is the passive movement of water across a semipermeable membrane.",                9),
    ("Chemical bonds form when atoms share or transfer electrons to achieve stability.",         9),
    ("Natural selection drives evolutionary change by favouring heritable advantageous traits.", 9),
    ("Radioactive decay reduces the number of unstable nuclei exponentially over time.",        9),
    ("The Krebs cycle generates electron carriers used in cellular respiration.",               9),
    ("Entropy in thermodynamics measures the degree of disorder in a system.",                  9),
    ("Speciation occurs when populations diverge sufficiently to become reproductively isolated.",9),

    # 10 — computer_science
    ("A hash map stores key-value pairs using a hash function for constant-time lookups.",      10),
    ("Gradient descent minimises a loss function by updating parameters in the negative direction.",10),
    ("Big-O notation describes the upper bound on the time complexity of an algorithm.",         10),
    ("Recursion solves problems by having functions call themselves on smaller sub-problems.",  10),
    ("A neural network learns representations by adjusting weights via backpropagation.",       10),
    ("Concurrency allows multiple tasks to make progress within overlapping time periods.",     10),
    ("A binary search tree maintains sorted data for efficient insertion and lookup.",          10),
    ("TCP/IP provides reliable ordered delivery of packets across networks.",                  10),
    ("A compiler translates high-level source code into machine-executable instructions.",      10),
    ("The PageRank algorithm ranks web pages by simulating random walks on a directed graph.",  10),

    # 11 — social_sciences
    ("Socialization is the process by which individuals learn cultural norms and values.",      11),
    ("Cognitive dissonance arises when beliefs and actions are inconsistent with each other.", 11),
    ("The Gini coefficient measures income inequality within a population.",                   11),
    ("Ethnography involves immersive observation to study cultural practices in context.",     11),
    ("Survey sampling uses statistical methods to infer population parameters.",               11),
    ("Maslow's hierarchy of needs ranks human motivations from physiological to self-actualisation.",11),
    ("Gender is a social construct shaped by cultural norms and historical context.",          11),
    ("Social capital refers to the networks and norms that enable collective action.",         11),
    ("Urbanisation is the process by which an increasing proportion of population lives in cities.",11),
    ("The prisoner's dilemma demonstrates why individuals may not cooperate even when beneficial.",11),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class Dataset:
    """
    Lightweight dataset container.

    Stores (text, label) pairs and supports:
      • Shuffling
      • Train / validation split
      • Iteration over batches
    """

    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = list(samples)

    @classmethod
    def from_synthetic(cls) -> 'Dataset':
        """Use the built-in synthetic domain dataset."""
        return cls(SYNTHETIC_DATA[:])

    @classmethod
    def from_csv(cls, path: str, text_col: int = 0, label_col: int = 1,
                 has_header: bool = True,
                 num_classes: Optional[int] = None) -> 'Dataset':
        """Load from a CSV file with text and integer label columns."""
        if num_classes is not None and (not isinstance(num_classes, int) or num_classes <= 0):
            raise ValueError(f"num_classes must be a positive int or None, got {num_classes!r}")
        samples = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) <= max(text_col, label_col):
                    continue
                try:
                    label = int(row[label_col].strip())
                    text  = row[text_col].strip()
                except (ValueError, IndexError):
                    continue
                if num_classes is not None and not (0 <= label < num_classes):
                    continue
                samples.append((text, label))
        return cls(samples)

    @classmethod
    def from_txt(cls, path: str,
                 num_classes: Optional[int] = None) -> 'Dataset':
        """Load from a two-column TSV: label<TAB>text."""
        if num_classes is not None and (not isinstance(num_classes, int) or num_classes <= 0):
            raise ValueError(f"num_classes must be a positive int or None, got {num_classes!r}")
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                label_str, text = line.split("\t", 1)
                try:
                    label = int(label_str.strip())
                except ValueError:
                    continue
                if num_classes is not None and not (0 <= label < num_classes):
                    continue
                samples.append((text.strip(), label))
        return cls(samples)

    def shuffle(self, seed: Optional[int] = None):
        rng = random.Random(seed)
        rng.shuffle(self.samples)
        return self

    def split(self, val_fraction: float = 0.1, *, seed: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        """Split into (shuffled) train and validation sets."""
        if len(self.samples) < 2:
            raise ValueError(
                f"split() requires at least 2 samples; dataset has {len(self.samples)}"
            )
        samples = self.samples[:]
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_val = int(round(len(samples) * val_fraction))
        n_val = max(1, min(n_val, len(samples) - 1))
        val   = Dataset(samples[:n_val])
        train = Dataset(samples[n_val:])
        return train, val

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader (simple sequential / shuffled iterator)
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Iterates over a Dataset one sample at a time (NanoTensor is not batched).
    Shuffles at the start of each epoch if shuffle=True.

    Yields: (text: str, label: int)
    """

    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self._rng    = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)
        for idx in indices:
            yield self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    def class_distribution(self) -> dict:
        """Count samples per class label."""
        dist: dict = {}
        for _, label in self.dataset.samples:
            dist[label] = dist.get(label, 0) + 1
        return dist
