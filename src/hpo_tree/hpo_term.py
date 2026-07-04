"""Utilities for representing Human Phenotype Ontology (HPO) terms as a tree.

This module defines the :class:`HumanPhenotypeOntology` class, a lightweight tree
node that mirrors a subset of the HPO graph as a rooted tree (using
``is_a``/superclass relationships). It also provides a convenience loader to
build the tree from an ontology source via :func:`src.hpo_tree.hpo_loader.load_hpo`.

Typical usage:

    >>> root = HumanPhenotypeTerm.load_ontology(output_dir="./data")
    >>> root.id
    'HP:0000001'
    >>> len(root.successors)  # direct children of the root term
    5  # example

Note: This structure keeps only a single parent per node to form a tree, even
though the underlying HPO is a DAG where a term may have multiple parents.
"""

from typing import Self, List, Optional

from loguru import logger

from src.hpo_tree.hpo_loader import load_hpo


class HumanPhenotypeTerm(object):
    """A node representing a single HPO term in a parent-children tree.

    This class stores basic metadata of an HPO term (``id``, ``name``,
    ``definition``, ``comment``) and maintains pointers to its predecessor
    (single parent) and successors (children). It provides helpers for tree
    navigation and a static loader to build a tree from the ontology.

    Limitation: While HPO is a DAG, this structure assigns at most one
    predecessor (parent) per node to keep a tree shape.
    """

    def __init__(self, id: str, name: str, definition: str, comment: str):
        """Create a new HPO term node.

        Args:
            id: HPO identifier (e.g., ``"HP:0000001"``).
            name: Human-readable term name.
            definition: The term's textual definition.
            comment: Optional comment associated with the term.
        """
        self._id = id
        self._name = name
        self._definition = definition
        self._comment = comment
        self._parent = None
        self._successors = []

    @property
    def id(self) -> str:
        """Return the HPO identifier for this term."""
        return self._id

    @property
    def name(self) -> str:
        """Return the human-readable name of this term."""
        return self._name

    @property
    def definition(self) -> str:
        """Return the textual definition of this term."""
        return self._definition

    @property
    def comment(self) -> str:
        """Return the comment associated with this term, if any."""
        return self._comment

    @property
    def predecessor(self) -> Self:
        """Return the parent node of this term, or ``None`` if it is the root."""
        return self._parent

    @property
    def successors(self) -> List[Self]:
        """Return the list of child nodes (direct successors)."""
        return self._successors

    @property
    def level(self) -> int:
        """Return the depth of this node within the tree.

        The root node has level ``0``. Each step downward increases the level
        by 1.
        """

        def recursive_parent_iter(parent, level):
            if parent is None:
                return level
            return recursive_parent_iter(parent.predecessor, level + 1)

        return 0 if self.predecessor is None else recursive_parent_iter(self.predecessor, 0)

    def add_successor(self, successor: Self) -> None:
        """Attach ``successor`` as a child of this node.

        This also sets the successor's parent pointer to this node.
        """
        if successor._parent is not None:
            successor._parent.remove_successor(successor)
        successor._parent = self
        self._successors.append(successor)
        self._successors.sort(key=lambda s: s.id)

    def delete(self) -> None:
        """
        Detaches this node from its parent's list of children.
        """
        if self.predecessor:
            self.predecessor.remove_successor(self)

    def remove_successor(self, successor: Self) -> None:
        """Detach ``successor`` from this node's children.

        This also clears the successor's parent pointer.
        """
        if successor in self.successors:
            self._successors.remove(successor)
            successor._parent = None

    def remove_successor_by_id(self, id: str):
        """
        Finds a direct successor by its HPO ID and detaches it from this node.

        Args:
            id (str): The HPO ID of the successor to remove.
        """
        successor = self.find_successor(id)
        if successor and successor.predecessor:
            successor.predecessor.remove_successor(successor)

    def clear_successors(self) -> None:
        """Remove all successors from this node.

        This also clears the successor's parent pointer.
        """
        for successor in self.successors:
            self.remove_successor(successor)

    def find_successor(self, id: str) -> Optional[Self]:
        """Find a descendant node by HPO id.

        Performs a depth-first search among this node's subtree.

        Args:
            id: The HPO identifier to search for.

        Returns:
            The matching descendant :class:`HumanPhenotypeTerm` or ``None`` if
            not found.
        """
        if self.id == id or self.name == id:
            return self
        for child in self.successors:
            successor = child.find_successor(id)
            if successor:
                return successor
        return None

    def find_predecessor(self, id: str) -> Optional[Self]:
        """Find the closest descendant whose parent has the given HPO id.

        Starting from this node, traverse upward through parents.

        Args:
            id: The HPO identifier of the ancestor to match.

        Returns:
            This node if its parent matches ``id``, otherwise the first node
            upward whose parent matches; ``None`` if no such ancestor exists.
        """
        if self.predecessor is None:
            return None
        if self.predecessor.id == id or self.predecessor.name == id:
            return self
        return self.predecessor.find_predecessor(id)

    def define_as_root(self):
        """
        Defines this term as a root node by clearing its parent pointer.
        """
        self._parent = None

    def find_root(self):
        """
        Traverses up the tree to find the root node.

        Returns:
            HumanPhenotypeTerm: The root node of the tree.
        """
        if self.predecessor is None:
            return self
        return self.predecessor.find_root()

    def is_leaf(self):
        """
        Checks if this term is a leaf node (has no successors).

        Returns:
            bool: True if it is a leaf, False otherwise.
        """
        return len(self) == 0

    def predecessors(self, with_self: bool = True) -> List[Self]:
        """
        Returns a list of all ancestor nodes (predecessors) of this term.

        Args:
            with_self (bool, optional): Whether to include this node in the result. Defaults to True.

        Returns:
            List[HumanPhenotypeTerm]: A list of ancestor nodes.
        """
        result = set()
        if with_self:
            result.update([self])
        if self.predecessor is not None:
            result.update(self.predecessor.predecessors(True))
        return list(result)

    def leafs(self) -> List[Self]:
        """
        Returns a list of all leaf nodes in the subtree rooted at this node.

        Returns:
            List[HumanPhenotypeTerm]: A list of descendant leaf nodes.
        """
        result = [self] if self.is_leaf() else []
        for child in self.successors:
            result.extend(child.leafs())
        return result

    def all_successors(self, with_self: bool = True) -> List[Self]:
        """
        Returns a list of all descendant nodes (successors) of this term recursively.

        Args:
            with_self (bool, optional): Whether to include this node in the result. Defaults to True.

        Returns:
            List[HumanPhenotypeTerm]: A list of all descendant nodes.
        """
        result = []
        if with_self:
            result.append(self)
        for child in self.successors:
            result.extend(child.all_successors(True))
        return result

    def __len__(self):
        """Return the size of the subtree rooted at this node.

        This counts all descendants recursively (i.e., the number of nodes in
        the subtree excluding the current node itself).
        """
        return len(self.successors) + sum([len(c) for c in self.successors])

    def __repr__(self):
        """Return a concise representation ``"<id>: <name>"``."""
        return f'{self.id} ({self.name})'

    @staticmethod
    def load_ontology(output_dir: str, download: bool = True) -> "HumanPhenotypeTerm":
        """Load HPO from disk (downloading if needed) and build a tree.

        The resulting structure is rooted at ``HP:0000001`` ("All"). Each term
        receives at most one parent to maintain a tree, using immediate
        superclasses from the ontology.

        Args:
            output_dir: Directory used to store/load ontology resources.
            download: If ``True``, download the ontology when not present.

        Returns:
            The root :class:`HumanPhenotypeTerm` corresponding to
            ``HP:0000001``.
        """
        hpo = load_hpo(output_dir=output_dir, download=download)
        logger.debug('Creating HPO tree structure...')
        hpterms = {}
        for term in hpo.terms():
            if not term.obsolete:
                hpterms.update({term.id: HumanPhenotypeTerm(id=term.id, name=term.name, definition=term.definition,
                                                            comment=term.comment)})

        for term in hpo.terms():
            for parent in term.superclasses(distance=1, with_self=False):
                if not parent.obsolete:
                    hpterms[parent.id].add_successor(hpterms[term.id])
                    break  # Use only the first available parent to be found!

        logger.debug('HPO tree created!')
        return hpterms['HP:0000001']
