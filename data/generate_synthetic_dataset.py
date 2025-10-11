"""
Generate synthetic equation-description pairs for pilot training.
Fast way to validate the full pipeline before scaling to real data.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physics equation templates with descriptions
EQUATION_TEMPLATES = [
    # Classical Mechanics
    {
        "latex": r"F = ma",
        "description": "Newton's second law of motion relating force, mass and acceleration",
        "domain": "classical_mechanics",
        "tree": {"root": "=", "left": "F", "right": {"op": "*", "left": "m", "right": "a"}}
    },
    {
        "latex": r"E_k = \frac{1}{2}mv^2",
        "description": "Kinetic energy equation for a moving object with mass and velocity",
        "domain": "classical_mechanics"
    },
    {
        "latex": r"p = mv",
        "description": "Linear momentum defined as the product of mass and velocity",
        "domain": "classical_mechanics"
    },
    # Electromagnetism
    {
        "latex": r"F = q(E + v \times B)",
        "description": "Lorentz force on a charged particle in electromagnetic field",
        "domain": "electromagnetism"
    },
    {
        "latex": r"\nabla \cdot E = \frac{\rho}{\epsilon_0}",
        "description": "Gauss's law relating electric field divergence to charge density",
        "domain": "electromagnetism"
    },
    {
        "latex": r"\nabla \times B = \mu_0 J + \mu_0\epsilon_0 \frac{\partial E}{\partial t}",
        "description": "Ampere-Maxwell law relating magnetic field curl to current and changing electric field",
        "domain": "electromagnetism"
    },
    # Quantum Mechanics
    {
        "latex": r"i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi",
        "description": "Schrödinger equation governing the time evolution of quantum states",
        "domain": "quantum_mechanics"
    },
    {
        "latex": r"[\hat{x}, \hat{p}] = i\hbar",
        "description": "Canonical commutation relation between position and momentum operators",
        "domain": "quantum_mechanics"
    },
    {
        "latex": r"\Delta x \Delta p \geq \frac{\hbar}{2}",
        "description": "Heisenberg uncertainty principle for position and momentum",
        "domain": "quantum_mechanics"
    },
    # Thermodynamics
    {
        "latex": r"dU = \delta Q - \delta W",
        "description": "First law of thermodynamics relating internal energy change to heat and work",
        "domain": "thermodynamics"
    },
    {
        "latex": r"S = k_B \ln \Omega",
        "description": "Boltzmann entropy formula relating entropy to number of microstates",
        "domain": "thermodynamics"
    },
    {
        "latex": r"PV = nRT",
        "description": "Ideal gas law relating pressure, volume, amount and temperature",
        "domain": "thermodynamics"
    },
    # Relativity
    {
        "latex": r"E = mc^2",
        "description": "Einstein's mass-energy equivalence equation",
        "domain": "relativity"
    },
    {
        "latex": r"ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2",
        "description": "Minkowski metric for flat spacetime in special relativity",
        "domain": "relativity"
    },
    {
        "latex": r"G_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}",
        "description": "Einstein field equations relating spacetime curvature to energy-momentum",
        "domain": "relativity"
    },
]

# Additional variations
EQUATION_VARIATIONS = {
    "wave_equation": [
        (r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u", 
         "Wave equation describing propagation of waves in a medium"),
        (r"\Box \phi = 0",
         "D'Alembert operator equation for massless wave propagation"),
    ],
    "schrodinger": [
        (r"-\frac{\hbar^2}{2m}\nabla^2\psi + V\psi = E\psi",
         "Time-independent Schrödinger equation for energy eigenstates"),
        (r"\hat{H}|\psi\rangle = E|\psi\rangle",
         "Eigenvalue equation for the Hamiltonian operator"),
    ],
    "maxwell": [
        (r"\nabla \times E = -\frac{\partial B}{\partial t}",
         "Faraday's law of electromagnetic induction"),
        (r"\nabla \cdot B = 0",
         "Gauss's law for magnetism showing no magnetic monopoles"),
    ],
}


def generate_synthetic_dataset(
    num_samples: int = 10000,
    train_ratio: float = 0.875,
    val_ratio: float = 0.0625
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate synthetic equation-description pairs.
    
    Args:
        num_samples: Total number of samples to generate
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    all_equations = EQUATION_TEMPLATES.copy()
    
    # Add variations
    for var_list in EQUATION_VARIATIONS.values():
        for latex, desc in var_list:
            all_equations.append({
                "latex": latex,
                "description": desc,
                "domain": "physics"
            })
    
    logger.info(f"Base equation templates: {len(all_equations)}")
    
    # Generate augmented samples
    dataset = []
    samples_per_eq = num_samples // len(all_equations)
    
    for eq_template in all_equations:
        for i in range(samples_per_eq):
            # Add slight variations to description
            desc = eq_template["description"]
            
            # Variation strategies
            variations = [
                desc,
                f"This equation represents {desc.lower()}",
                f"The formula {desc}",
                f"Mathematical expression for {desc.lower()}",
            ]
            
            sample = {
                "id": f"syn_{len(dataset):06d}",
                "equation": eq_template["latex"],
                "description": random.choice(variations),
                "domain": eq_template.get("domain", "physics"),
                "metadata": {
                    "source": "synthetic",
                    "template_id": all_equations.index(eq_template)
                }
            }
            dataset.append(sample)
    
    # Shuffle
    random.shuffle(dataset)
    
    # Split
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]
    
    logger.info(f"Generated {len(dataset)} total samples")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def save_dataset(train_data: List[Dict], val_data: List[Dict], 
                test_data: List[Dict], output_dir: str = "./data/dataset"):
    """Save dataset splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_path / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Dataset saved to {output_path}")
    
    # Save dataset info
    info = {
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "total": len(train_data) + len(val_data) + len(test_data),
        "domains": list(set(d["domain"] for d in train_data)),
        "source": "synthetic"
    }
    
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info: {info}")


def main():
    """Generate and save synthetic dataset."""
    logger.info("Generating synthetic equation-description dataset...")
    
    # Generate 10K samples for pilot
    train_data, val_data, test_data = generate_synthetic_dataset(num_samples=10000)
    
    # Save
    save_dataset(train_data, val_data, test_data)
    
    logger.info("✓ Synthetic dataset generation complete!")
    logger.info("Next step: Run training with: python training/train.py")


if __name__ == "__main__":
    random.seed(42)
    main()
