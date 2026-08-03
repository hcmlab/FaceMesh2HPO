import {PredictionResult} from "@/utils/PredictionResult";

export interface TreeNode {
    result: PredictionResult;
    children: TreeNode[];
    level: number;
}

export const buildHierarchy = (results: PredictionResult[]): TreeNode[] => {
    // ID → Result Map
    const resultMap = new Map(results.map(r => [r.id, r]));

    // Root-Level (parent === null)
    const roots: TreeNode[] = [];

    results.forEach(result => {
        if (!result.parent) {
            roots.push(buildTreeNode(result, resultMap, 0));
        }
    });

    // Sort roots: Positive > Negative > Probability
    return roots.sort((a, b) => {
        if (a.result.isPositive !== b.result.isPositive) {
            return b.result.isPositive ? 1 : -1;
        }
        return b.result.probability - a.result.probability;
    });
};

const buildTreeNode = (
    result: PredictionResult,
    resultMap: Map<string, PredictionResult>,
    level: number
): TreeNode => {
    const children = Array.from(resultMap.values())
        .filter(r => r.parent === result.id)
        .map(childResult => buildTreeNode(childResult, resultMap, level + 1));

    return {
        result,
        children,
        level
    };
};

export const flattenTree = (tree: TreeNode[]): PredictionResult[] => {
    const flat: PredictionResult[] = [];

    const recurse = (node: TreeNode) => {
        flat.push(node.result);
        node.children.forEach(recurse);
    };

    tree.forEach(recurse);
    return flat;
};

export const findNodeById = (tree: TreeNode[], id: string): TreeNode | null => {
    for (const node of tree) {
        if (node.result.id === id) return node;
        const found = findNodeById(node.children, id);
        if (found) return found;
    }
    return null;
};