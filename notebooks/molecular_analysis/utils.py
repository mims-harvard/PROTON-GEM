"""Shared utilities for molecular analysis notebooks.

This module provides common setup and data loading functions used across
the molecular analysis notebooks, reducing code duplication.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import torch

if TYPE_CHECKING:
    from logging import Logger

    import dgl


def find_repo_root(start: Path | None = None) -> Path:
    """Find PROTON-GEM root by walking up to `pyproject.toml`.

    Args:
        start: Starting directory. Defaults to current working directory.

    Returns:
        Path to repository root.
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


def setup_notebook_environment(logger_name: str = "notebook") -> tuple[Path, Logger]:
    """Standard setup for molecular analysis notebooks.

    This function:
    1. Finds and changes to the repository root directory
    2. Adds the root to sys.path for imports
    3. Loads environment variables from secrets.env
    4. Configures matplotlib with Arial font if available
    5. Sets up logging

    Args:
        logger_name: Name for the logger instance.

    Returns:
        Tuple of (PROJECT_ROOT path, configured logger).
    """
    # Find and set project root
    project_root = find_repo_root()
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import config after path setup
    from src.config import conf

    # Load environment variables
    try:
        from dotenv import load_dotenv

        if load_dotenv is not None:
            load_dotenv(conf.paths.secrets_path)
    except ImportError:
        pass

    # Configure matplotlib
    if any("Arial" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(logger_name)

    logger.info(f"Working from: {project_root}")

    return project_root, logger


def load_kg_and_model(
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dgl.DGLGraph, object, torch.Tensor]:
    """Load knowledge graph data, PROTON model, and embeddings.

    Args:
        device: Torch device for model. Defaults to auto-detected device.

    Returns:
        Tuple of (nodes_df, edges_df, kg_graph, proton_model, embeddings).
    """
    from src.config import conf
    from src.constants import TORCH_DEVICE
    from src.dataloaders import load_graph
    from src.models import HGT

    if device is None:
        device = TORCH_DEVICE

    # Load nodes and edges
    nodes = pd.read_csv(
        conf.paths.kg.nodes_path,
        dtype={"node_index": int},
        low_memory=False,
    )
    edges = pd.read_csv(
        conf.paths.kg.edges_path,
        dtype={"edge_index": int, "x_index": int, "y_index": int},
        low_memory=False,
    )

    # Build graph
    kg = load_graph(nodes, edges)

    # Load model
    model = HGT.load_from_checkpoint(
        checkpoint_path=str(conf.paths.checkpoint.checkpoint_path),
        kg=kg,
        strict=False,
    )
    model.eval()
    model = model.to(device)

    # Load embeddings
    embeddings = torch.load(
        conf.paths.checkpoint.embeddings_path,
        map_location="cpu",
    )

    return nodes, edges, kg, model, embeddings


def get_output_paths() -> tuple[Path, Path, Path, Path]:
    """Get standard output paths from config.

    Returns:
        Tuple of (OUTPUT_DIR, SMILES_CACHE_PATH, UNIMOL_CACHE_PATH, ADAPTER_PATH).
    """
    from src.config import conf

    output_dir = conf.paths.notebooks.base_dir / conf.paths.notebooks.molecular_analysis.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_cache = conf.paths.kg.base_dir / conf.paths.kg.smiles_cache
    unimol_cache = output_dir / conf.paths.notebooks.molecular_analysis.unimol_cache
    adapter_path = conf.paths.checkpoint.base_dir / conf.paths.checkpoint.molecular_adapter

    return output_dir, smiles_cache, unimol_cache, adapter_path


def get_inchi_key(smiles: str) -> str | None:
    """Convert SMILES string to InChIKey for deduplication.

    Args:
        smiles: SMILES string representation of molecule.

    Returns:
        InChIKey string or None if conversion fails.
    """
    if not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol) if mol else None
    except Exception:
        return None


def filter_by_node_type(nodes: pd.DataFrame, node_type: str) -> pd.DataFrame:
    """Filter nodes DataFrame by node type.

    Args:
        nodes: DataFrame with 'node_type' column.
        node_type: Type to filter for (e.g., 'drug', 'gene/protein', 'disease').

    Returns:
        Filtered DataFrame with reset index.
    """
    return nodes[nodes["node_type"] == node_type].copy().reset_index(drop=True)
