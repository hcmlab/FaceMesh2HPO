import {TreeNode} from "@/utils/TreeNode";
import {HPONode} from "@/components/HPONode";
import {PredictionResult} from "@/utils/PredictionResult";
import React, {useEffect, useMemo} from "react";

interface HPOTreeProps {
    nodes: TreeNode[];
    searchTerm: string;
    minConfidence: number;
    expandedId: string | null;
    onSelect: (node: PredictionResult | null) => void;
    onToggleExpand: (id: string) => void;
    onFilterStatsChange?: (stats: { visibleCount: number; matchCount: number }) => void;
}

export const HPOTree = ({
                            nodes,
                            searchTerm,
                            minConfidence,
                            expandedId,
                            onSelect,
                            onToggleExpand,
                            onFilterStatsChange,
                        }: HPOTreeProps) => {
    const confidenceFilteredNodes = useMemo(() => {
        const pruneByConfidence = (tree: TreeNode[]): TreeNode[] =>
            tree.reduce<TreeNode[]>((acc, node) => {
                const passesConfidence = node.result.probability >= minConfidence;
                const prunedChildren = pruneByConfidence(node.children);

                if (!passesConfidence && prunedChildren.length === 0) return acc;

                acc.push({
                    ...node,
                    children: prunedChildren,
                });

                return acc;
            }, []);

        return pruneByConfidence(nodes);
    }, [nodes, minConfidence]);

    const filteredNodes = useMemo(() => {
        const lowerSearch = searchTerm.trim().toLowerCase();

        if (!lowerSearch) return confidenceFilteredNodes;

        const pruneBySearch = (tree: TreeNode[]): TreeNode[] =>
            tree.reduce<TreeNode[]>((acc, node) => {
                const matches =
                    node.result.name.toLowerCase().includes(lowerSearch) ||
                    node.result.hpoId?.toLowerCase().includes(lowerSearch);

                const prunedChildren = pruneBySearch(node.children);

                if (!matches && prunedChildren.length === 0) return acc;

                acc.push({
                    ...node,
                    children: prunedChildren,
                });

                return acc;
            }, []);

        return pruneBySearch(confidenceFilteredNodes);
    }, [confidenceFilteredNodes, searchTerm]);

    const visibleCount = useMemo(() => {
        const countVisible = (tree: TreeNode[]): number =>
            tree.reduce((count, node) => count + 1 + countVisible(node.children), 0);

        return countVisible(filteredNodes);
    }, [filteredNodes]);

    const matchCount = useMemo(() => {
        const lowerSearch = searchTerm.trim().toLowerCase();

        if (!lowerSearch) return visibleCount;

        const countMatches = (tree: TreeNode[]): number =>
            tree.reduce((count, node) => {
                const matches =
                    node.result.name.toLowerCase().includes(lowerSearch) ||
                    node.result.hpoId?.toLowerCase().includes(lowerSearch);

                return count + (matches ? 1 : 0) + countMatches(node.children);
            }, 0);

        return countMatches(filteredNodes);
    }, [filteredNodes, searchTerm, visibleCount]);

    useEffect(() => {
        onFilterStatsChange?.({
            visibleCount,
            matchCount,
        });
    }, [visibleCount, matchCount, onFilterStatsChange]);

    return (
    <div className="hpo-tree">
      <style jsx global>{`
        .hpo-tree {
          display: flex;
          flex-direction: column;
          gap: 0.01rem;
          width: 100%;
        }

        .hpo-tree-root-group,
        .hpo-children {
          display: flex;
          flex-direction: column;
          gap: 0.01rem;
          width: 100%;
          min-width: 0;
        }

        .hpo-tree-item {
          display: flex;
          flex-direction: column;
          gap: 0.01rem;
          width: 100%;
          min-width: 0;
        }

        .hpo-tree-node-wrap {
          width: 100%;
          min-width: 0;
        }

        .hpo-children {
          margin-left: 0.1rem;
          padding-left: 0.85rem;
          border-left: 1px solid rgba(148, 163, 184, 0.14);
        }

        .hpo-tree-item[data-expanded="true"] > .hpo-children {
          margin-top: 0.05rem;
        }

        @media (max-width: 768px) {
          .hpo-tree {
            gap: 0.01rem;
          }

          .hpo-tree-root-group,
          .hpo-children {
            gap: 0.01rem;
          }

          .hpo-tree-item {
            gap: 0.01rem;
          }

          .hpo-children {
            margin-left: 0.1rem;
            padding-left: 0.1rem;
          }
        }
      `}</style>

      <div className="hpo-tree-root-group">
        {filteredNodes.map((node) => (
          <TreeNodeView
            key={node.result.id}
            node={node}
            expandedId={expandedId}
            onSelect={onSelect}
            onToggleExpand={onToggleExpand}
          />
        ))}
      </div>
    </div>
  );
};

interface TreeNodeViewProps {
    node: TreeNode;
    expandedId: string | null;
    onSelect: (node: PredictionResult | null) => void;
    onToggleExpand: (id: string) => void;
}

const TreeNodeView = ({
                          node,
                          expandedId,
                          onSelect,
                          onToggleExpand,
                      }: TreeNodeViewProps) => {
    const {result, children} = node;
    const isExpanded = expandedId === result.id;

    const sortedChildren = [...children].sort((a, b) =>
        b.result.isPositive === a.result.isPositive
            ? b.result.probability - a.result.probability
            : b.result.isPositive
                ? 1
                : -1
    );

    return (
        <div className="hpo-tree-item" data-expanded={isExpanded ? "true" : "false"}>
            <div className="hpo-tree-node-wrap">
                <HPONode
                    node={result}
                    isExpanded={isExpanded}
                    onSelect={onSelect}
                    onToggleExpand={() => onToggleExpand(result.id)}
                />
            </div>

            {sortedChildren.length > 0 && (
                <div className="hpo-children">
                    {sortedChildren.map((child) => (
                        <TreeNodeView
                            key={child.result.id}
                            node={child}
                            expandedId={expandedId}
                            onSelect={onSelect}
                            onToggleExpand={onToggleExpand}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};