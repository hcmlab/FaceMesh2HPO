import {HPOModel} from "@/utils/HPOModel";

export interface PredictionResult {
    id: string;
    name: string;
    description: string;
    probability: number;
    isPositive: boolean;
    hpoId: string;
    parent: string | null;
    metrics?: HPOModel["metrics"];
    database?: HPOModel["database"];
    syndromes?: HPOModel["syndromes"];
    shapValues?: number[];
}