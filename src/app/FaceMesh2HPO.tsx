"use client";

import React, {useCallback, useEffect, useRef, useState} from "react";
import Script from "next/script";
import {KernelSHAP} from "webshap";
import {PredictionResult} from "@/utils/PredictionResult";
import {HPOModel, HPOModelCalibration} from "@/utils/HPOModel";
import {buildHierarchy, TreeNode} from "@/utils/TreeNode";
import {HPOTree} from "@/components/HPOTree";
import {PatientSetupFlow, PatientSetupSubmitPayload,} from "@/components/PatientSetupFlow";
import {AppInitializationProgress} from "@/components/AppInitializationProgress";

declare global {
    interface Window {
        FaceMesh: any;
        drawConnectors: any;
        drawLandmarks: any;
        FACEMESH_TESSELATION: any;
        ort: any;
        Plotly: any;
    }
}

// const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const BASE_PATH = "https://raw.githubusercontent.com/hcmlab/FaceMesh2HPOWebApp/refs/heads/master/public";
const REFERENCE_MESH_DATA = `${BASE_PATH}/models/result.json`;

function prepareDatabase(database: any) {
    const gender: { [key: string]: number } = {};
    const ethnicity: { [key: string]: number } = {};
    const ageGroups: { [key: string]: number } = {};
    const databaseSource: { [key: string]: number } = {};

    const genderKeys = ["female", "male", "divers", "unknown"];
    const ethnicityKeys = ["African", "Asian", "European", "Others", "Unknown"];
    const sourceKeys = ["GMDB", "UTK"];

    Object.entries(database).forEach(([key, val]: [string, any]) => {
        const count = val["0"] || 0;

        if (genderKeys.includes(key)) {
            gender[key] = count;
        } else if (ethnicityKeys.includes(key)) {
            ethnicity[key] = count;
        } else if (sourceKeys.includes(key)) {
            databaseSource[key] = count;
        } else {
            if (!isNaN(Number(key)) || key === "Missing") {
                const age = key === "Missing" ? 0 : Number(key);
                let group = "Unknown";

                if (key === "Missing" || age === 0) {
                    group = "Unknown";
                } else if (age >= 1 && age <= 5) {
                    group = "1-5";
                } else if (age >= 6 && age <= 10) {
                    group = "6-10";
                } else {
                    group = "10+";
                }

                ageGroups[group] = group in ageGroups ? ageGroups[group] + count : count;
            }
        }
    });

    return {gender, ethnicity, ageGroups, databaseSource};
}

export default function FaceMesh2HPO() {
    const [statusText, setStatusText] = useState("");
    const [modelStatus, setModelStatus] = useState("Initializing...");
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [loadingETA, setLoadingETA] = useState<string | null>(null);
    const [results, setResults] = useState<PredictionResult[]>([]);
    const [isPredicting, setIsPredicting] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [expandedId, setExpandedId] = useState<string | null>(null);
    const [selectedHPO, setSelectedHPO] = useState<string | null>(null);
    const [hierarchy, setHierarchy] = useState<TreeNode[]>([]);
    const [isUploading, setIsUploading] = useState(false);

    const [showAppInitialization, setShowAppInitialization] = useState(false);
    const [showSetupFlow, setShowSetupFlow] = useState(false);
    const [showProcessing, setShowProcessing] = useState(false);
    const [showMainContent, setShowMainContent] = useState(false);
    const [showColorLegend, setShowColorLegend] = useState(false);
    const [showNoFaceModal, setShowNoFaceModal] = useState(false);
    const [filterStats, setFilterStats] = useState({visibleCount: 0, matchCount: 0});

    const [age, setAge] = useState<number>(0);
    const [gender, setGender] = useState<number>(-1);
    const [ethnicity, setEthnicity] = useState<number>(-1);
    const [minConfidence, setMinConfidence] = useState<number>(0);

    const faceCanvasRef = useRef<HTMLCanvasElement>(null);
    const faceMeshDetectorRef = useRef<any>(null);
    const sessionsRef = useRef<Map<string, { session: any; config: HPOModel }>>(
        new Map()
    );
    const imageRef = useRef<HTMLImageElement | null>(null);
    const hpoModelsRef = useRef<HPOModel[]>([]);
    const landmarksRef = useRef<any[]>([]);
    const referenceLandmarksRef = useRef<number[][]>([]);
    const pendingSetupImageRef = useRef<HTMLImageElement | null>(null);
    const pendingFaceDetectionRef = useRef(false);

    let colorScaleLegend: [number, string][] = [
        [0, "#3939f6"],
        [0.01, "#bda87b"],
        [1, "#dd0303"],
    ];

    async function decompressGzip(buffer: ArrayBuffer): Promise<ArrayBuffer> {
        const ds = new DecompressionStream("gzip");
        const stream = new Blob([buffer]).stream().pipeThrough(ds);
        const decompressedResponse = new Response(stream);
        return await decompressedResponse.arrayBuffer();
    }

    async function loadCompressedOnnxModel(url: string, options: any) {
        const response = await fetch(url);
        const compressedBytes = await response.arrayBuffer();
        const rawBytes = await decompressGzip(compressedBytes);
        return await window.ort.InferenceSession.create(rawBytes, options);
    }

    function parseCalibration(raw: any): HPOModelCalibration | undefined {
        if (!raw || typeof raw !== "object" || typeof raw.method !== "string") {
            return undefined;
        }

        const base = {
            method: raw.method,
            n: Number(raw.n ?? 0),
            n_pos: Number(raw.n_pos ?? 0),
            brier_raw: Number(raw.brier_raw ?? 0),
            brier_cal: Number(raw.brier_cal ?? 0),
            ece_raw: Number(raw.ece_raw ?? 0),
            ece_cal: Number(raw.ece_cal ?? 0),
        };

        switch (raw.method) {
            case "beta":
                if (
                    typeof raw.a === "number" &&
                    typeof raw.b === "number" &&
                    typeof raw.c === "number"
                ) {
                    return {
                        ...base,
                        method: "beta",
                        a: raw.a,
                        b: raw.b,
                        c: raw.c,
                    };
                }
                return undefined;

            case "temperature":
                if (typeof raw.T === "number") {
                    return {
                        ...base,
                        method: "temperature",
                        T: raw.T,
                    };
                }
                return undefined;

            case "none":
                return {
                    ...base,
                    method: "none",
                };

            default:
                return undefined;
        }
    }

    const initModels = async () => {
        setShowAppInitialization(true);

        try {
            const response = await fetch(REFERENCE_MESH_DATA);
            const data = await response.json();

            referenceLandmarksRef.current = data.reference_mesh.map((item: {
                id: number;
                x: number;
                y: number;
                z: number;
            }) => [item.x, item.y, item.z]);

            const models: HPOModel[] = data.nodes.map((node: any) => {
                let bestFold = "0";
                let maxAuroc = -1;

                colorScaleLegend = [
                    [0, "#3939f6"],
                    [node.feature_importance_threshold, "#bda87b"],
                    [1, "#dd0303"],
                ];

                Object.entries(node.metrics.auroc).forEach(([fold, val]) => {
                    if ((val as number) > maxAuroc) {
                        maxAuroc = val as number;
                        bestFold = fold;
                    }
                });

                const m = node.metrics;
                const tp = m["stat_scores/tp"]?.[bestFold] || 0;
                const fp = m["stat_scores/fp"]?.[bestFold] || 0;
                const tn = m["stat_scores/tn"]?.[bestFold] || 0;
                const fn = m["stat_scores/fn"]?.[bestFold] || 0;

                const bestMetrics = {
                    accuracy: m.accuracy?.[bestFold] || 0,
                    auroc: m.auroc?.[bestFold] || 0,
                    f1: m.f1_score?.[bestFold] || 0,
                    precision: m.precision?.[bestFold] || 0,
                    recall: m.recall?.[bestFold] || 0,
                    specificity: tn + fp > 0 ? tn / (tn + fp) : 0,
                    validation_samples: tp + fp + tn + fn,
                    tp,
                    fp,
                    tn,
                    fn,
                };

                const databaseInfo = {
                    affected: prepareDatabase(node.database.hpo_present),
                    unaffected: prepareDatabase(node.database.hpo_absent),
                };

                const syndromesList = Object.entries(
                    node.database.syndromes.internal_syndrome_name
                )
                    .map(([key, name]) => ({
                        name: name as string,
                        count: node.database.syndromes.count?.[key] || 0,
                    }))
                    .filter((s) => s.count > 0)
                    .sort((a, b) => b.count - a.count);

                const importanceValues = node.importance_values
                    ? Object.fromEntries(
                        Object.entries(node.importance_values).map(([k, v]) => [k, (v as any)[0]])
                    )
                    : {};

                const calibration = parseCalibration(node.calibration);

                return {
                    id: node.id,
                    name: node.description,
                    description: node.definition,
                    url: `${BASE_PATH}/models/${node.id.replace(":", "_")}.onnx.gzip`,
                    parent: node.parent,
                    metaData: node.meta_data || [],
                    importanceValues,
                    feature_importance_threshold: node.feature_importance_threshold,
                    metrics: bestMetrics,
                    database: databaseInfo,
                    syndromes: syndromesList,
                    calibration,
                };
            });

            hpoModelsRef.current = models;

            if (!window.ort) {
                console.warn("Cannot load models: ORT not ready");
                return;
            }

            setModelStatus("Loading models...");

            const totalModels = models.length;
            let loadedCount = 0;
            const startTime = Date.now();

            for (const model of models) {
                if (sessionsRef.current.has(model.id)) {
                    loadedCount++;
                    continue;
                }

                try {
                    const sessionOptions = {
                        executionProviders: ["wasm", "cpu"],
                    };
                    const session = await loadCompressedOnnxModel(model.url, sessionOptions);
                    sessionsRef.current.set(model.id, {session, config: model});
                    loadedCount++;
                } catch (error) {
                    console.error(`Failed to load ${model.name} (${model.id}) at ${model.url}:`, error);
                }

                const progress = (loadedCount / totalModels) * 100;
                setLoadingProgress(progress);

                const elapsed = (Date.now() - startTime) / 1000;
                const avgTimePerModel = elapsed / loadedCount;
                const remainingModels = totalModels - loadedCount;
                const remainingSeconds = Math.round(remainingModels * avgTimePerModel);

                if (remainingSeconds > 60) {
                    setLoadingETA(
                        `${Math.floor(remainingSeconds / 60)}m ${remainingSeconds % 60}s remaining`
                    );
                } else {
                    setLoadingETA(`${remainingSeconds}s remaining`);
                }

                setModelStatus(`Loading models (${loadedCount}/${totalModels})...`);
            }

            if (window.FaceMesh && !faceMeshDetectorRef.current) {
                const faceMeshDetector = new window.FaceMesh({
                    locateFile: (file: string) =>
                        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
                });

                faceMeshDetector.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.8,
                    minTrackingConfidence: 0.8,
                    baseOptions: {
                        delegate: "CPU",
                        enableCpuFallback: false,
                    },
                    runningMode: "IMAGE",
                });

                faceMeshDetector.onResults(onFaceMeshResults);
                faceMeshDetectorRef.current = faceMeshDetector;
            }

            setModelStatus(`${sessionsRef.current.size} models loaded`);
            setLoadingProgress(100);
            setLoadingETA(null);
            setShowAppInitialization(false);
            setShowSetupFlow(true);
        } catch (error) {
            console.error("Failed to load HPO structure:", error);
            setModelStatus("Error loading models");
            setShowAppInitialization(false);
        }
    };

    const interpolateColor = (
        value: number,
        colorScale: [number, string][],
        defaultColor: string = "#efefef"
    ): string => {
        const clampedValue = Math.max(0, Math.min(1, value));

        for (let i = 0; i < colorScale.length - 1; i++) {
            const [low, lowColor] = colorScale[i];
            const [high, highColor] = colorScale[i + 1];

            if (clampedValue >= low && clampedValue <= high) {
                const ratio = (clampedValue - low) / (high - low);
                return interpolateRGB(lowColor, highColor, ratio);
            }
        }

        return defaultColor;
    };

    const interpolateRGB = (color1: string, color2: string, ratio: number): string => {
        const rgb1 = hexToRgb(color1);
        const rgb2 = hexToRgb(color2);

        if (!rgb1 || !rgb2) return color1;

        const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * ratio);
        const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * ratio);
        const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * ratio);

        return `rgb(${r}, ${g}, ${b})`;
    };

    const hexToRgb = (
        hex: string
    ): { r: number; g: number; b: number } | null => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
            ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16),
            }
            : null;
    };

    const getPointMaskFromImportance = (
        importanceValues?: Record<string, number>
    ): number[] => {
        if (!importanceValues) return [];

        return Object.keys(importanceValues)
            .map((k) => Number(k))
            .filter((v) => Number.isInteger(v) && v >= 0)
            .sort((a, b) => a - b);
    };

    const getRelevantPoints = (
        allPoints: number[][],
        pointMask: number[]
    ): number[][] => {
        return pointMask.map((idx) => {
            const p = allPoints[idx];
            if (!p) throw new Error(`landmark index ${idx} out of range`);
            return p;
        });
    };

    const redrawCanvas = useCallback((node: PredictionResult | null) => {
        if (!landmarksRef.current.length || !imageRef.current) return;

        const landmarks = landmarksRef.current;
        let shapValues: number[] | undefined;
        const model = hpoModelsRef.current.find((m) => m.id === node?.id);

        if ((node as any)?.shapValues) {
            shapValues = (node as any).shapValues;
        }

        const canvas = faceCanvasRef.current;
        if (!canvas || !imageRef.current) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = imageRef.current.width;
        canvas.height = imageRef.current.height;
        ctx.drawImage(imageRef.current, 0, 0);
        const minCanvasSize = Math.min(canvas.width, canvas.height);
        // Scale point radius with canvas size:
        // - 0.25% of the smaller canvas dimension
        // - clamped so points do not become too tiny or too large
        const radius = Math.max(1, minCanvasSize * 0.0025);

        if (window.drawLandmarks && landmarks && landmarks.length > 0) {
            const pointMask = getPointMaskFromImportance(model?.importanceValues);
            const pointsToDraw =
                pointMask.length > 0
                    ? pointMask.map((idx) => landmarks[idx]).filter(Boolean)
                    : landmarks;

            if (model) {
                const fallbackImportance = pointMask.map(
                    (idx) => {
                        // @ts-ignore
                        return model.importanceValues?.[String(idx)] ?? 0;
                    }
                );

                const values =
                    shapValues && shapValues.length === pointsToDraw.length
                        ? shapValues
                        : fallbackImportance;

                const absValues = values.map(Math.abs);
                const maxImportance = Math.max(...absValues, 1e-12);

                pointsToDraw.forEach((pt: any, i: number) => {
                    const valueImportance = values[i] ?? 0;
                    const normalizedImportance = Math.abs(valueImportance) / maxImportance;
                    const color = interpolateColor(normalizedImportance, colorScaleLegend);

                    ctx.beginPath();
                    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, radius, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                });
            } else {
                window.drawLandmarks(ctx, pointsToDraw, {
                    color: "rgb(194 235 255)",
                    lineWidth: 0.5,
                    radius: radius,
                });
            }
        }
    }, []);

    useEffect(() => {
        if (results.length > 0) {
            setHierarchy(buildHierarchy(results));
        } else {
            setHierarchy([]);
        }
    }, [results]);

    const handleHPONodeSelect = useCallback(
        (node: PredictionResult | null) => {
            redrawCanvas(node);
        },
        [redrawCanvas]
    );

    const runPredictions = async (landmarks: any[]) => {
        setIsPredicting(true);

        const t0 = performance.now();
        const rawPoints = landmarks.map((lm) => [lm.x, lm.y, lm.z]);
        const modelResults = [];

        for (const [modelId, sessionData] of Array.from(sessionsRef.current.entries())) {
            const {session, config} = sessionData;
            try {
                modelResults.push(await predictSingleModel(config, session, rawPoints));
            } catch (e) {
                console.error(`Model ${modelId} failed:`, e);
            }
        }

        const t1 = performance.now();
        const elapsedMs = t1 - t0;
        setStatusText(`Inference took ${elapsedMs.toFixed(1)} ms`);

        redrawCanvas(null);
        setResults(modelResults.filter(Boolean) as PredictionResult[]);
        setIsPredicting(false);
    };

    function sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }

    function calibrateProbability(logit: number, calibration?: HPOModelCalibration): number {
        if (!calibration || calibration.method === "none") {
            return sigmoid(logit);
        }

        if (calibration.method === "temperature") {
            return sigmoid(logit / calibration.T);
        }

        const eps = 1e-6;
        const p = Math.min(Math.max(sigmoid(logit), eps), 1 - eps);
        const z =
            calibration.a * Math.log(p) -
            calibration.b * Math.log(1 - p) +
            calibration.c;

        return sigmoid(z);
    }

    const predictSingleModel = async (
        config: HPOModel,
        session: any,
        alignedPoints: number[][]
    ): Promise<PredictionResult> => {
        const pointMask = getPointMaskFromImportance(config.importanceValues);
        const relevantPoints =
            pointMask.length > 0 ? getRelevantPoints(alignedPoints, pointMask) : alignedPoints;

        const nPoints = relevantPoints.length;
        const batch = 1;

        const inputData = packLandmarks(relevantPoints, nPoints, batch);
        const tensor = new window.ort.Tensor("float32", inputData, [batch, 3, nPoints]);
        const feeds = buildFeeds(tensor, config, batch);

        const resultsData = await session.run(feeds);
        const predictionRaw = resultsData.logits.data[0] as number;
        const predictionSigmoid = calibrateProbability(predictionRaw, config.calibration);
        const shapValues: number[] = [];

        return {
            id: config.id,
            name: config.name,
            description: config.description,
            probability: predictionSigmoid,
            isPositive: predictionSigmoid > 0.5,
            shapValues,
            hpoId: config.id,
            parent: config.parent,
            metrics: config.metrics,
            database: config.database,
            syndromes: config.syndromes,
        };
    };

    const packLandmarks = (
        points: number[][],
        N: number,
        batch: number
    ): Float32Array => {
        const channels = 3;
        const nPoints = N;
        const data = new Float32Array(batch * channels * nPoints);
        for (let c = 0; c < channels; c++) {
            for (let j = 0; j < nPoints; j++) {
                data[c * nPoints + j] = points[j][c];
            }
        }
        return data;
    };

    const buildFeeds = (tensor: any, config: HPOModel, batch: number = 1): any => {
        const feeds: any = {input: tensor};

        if (config.metaData?.length) {
            config.metaData.forEach((meta) => {
                let val = 0;
                if (meta.toLowerCase().includes("age")) val = age;
                else if (meta.toLowerCase().includes("gender")) val = gender;
                else if (meta.toLowerCase().includes("ethnicity")) val = ethnicity;

                const metaArr = new Float32Array(batch);
                metaArr.fill(val);
                feeds[meta] = new window.ort.Tensor("float32", metaArr, [batch]);
            });
        }

        return feeds;
    };

    const computeSHAP = async (
        session: any,
        inputData: Float32Array,
        baseFeeds: any,
        config: HPOModel,
        nSamples: number = 32
    ): Promise<number[] | undefined> => {
        if (config.id.includes("test")) return undefined;

        try {
            const predictFunc = (X: number[][]) => predictShapBatch(session, X, baseFeeds);
            const background = [Array.from(inputData.map((v) => Math.abs(v) * 0.5))];

            const explainer = new KernelSHAP(predictFunc, background, 0.42);
            const shapResult = await explainer.explainOneInstance(Array.from(inputData), nSamples);
            return shapResult[0];
        } catch (e) {
            console.warn("SHAP skipped:", e);
            return undefined;
        }
    };

    const predictShapBatch = async (
        session: any,
        X: number[][],
        baseFeeds: any
    ): Promise<number[][]> => {
        const predictions: number[][] = [];
        const N_rel = baseFeeds.input.dims[2];
        const BATCH = 2;

        for (const sample of X) {
            const perBatch = sample.length;
            const sampleData = new Float32Array(BATCH * perBatch);
            for (let b = 0; b < BATCH; b++) sampleData.set(sample, b * perBatch);

            const sampleTensor = new window.ort.Tensor("float32", sampleData, [
                BATCH,
                3,
                N_rel,
            ]);

            const sampleFeeds = {...baseFeeds, input: sampleTensor};
            const out = await session.run(sampleFeeds);
            const rowSize = out.logits.data.length / BATCH;
            predictions.push(Array.from(out.logits.data.slice(0, rowSize)));
        }

        return predictions;
    };

    const onFaceMeshResults = async (results: any) => {
        const canvas = results.image;
        if (!canvas) {
            console.warn("canvas not ready yet");
            return;
        }
        if (!imageRef.current) {
            console.warn("imageRef not ready yet");
            return;
        }

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            landmarksRef.current = [...landmarks];
            console.log("landmarks", landmarksRef.current);

            // setStatusText(`Face detected (${landmarks.length} landmarks)`);

            if (pendingFaceDetectionRef.current) {
                setShowSetupFlow(false);
                setShowProcessing(false);
                setShowMainContent(true);
                setShowNoFaceModal(false);
                pendingFaceDetectionRef.current = false;
            }

            if (!isPredicting) {
                runPredictions(landmarks);
            }
        } else {
            landmarksRef.current = [];
            setStatusText("No face detected");
            setShowProcessing(false);

            if (pendingFaceDetectionRef.current) {
                setShowNoFaceModal(true);
                setShowSetupFlow(true);
                setShowMainContent(false);
                imageRef.current = null;
                pendingSetupImageRef.current = null;
                pendingFaceDetectionRef.current = false;

                if (canvas) {
                    const ctx = canvas.getContext("2d");
                    ctx?.clearRect(0, 0, canvas.width, canvas.height);
                }
            }
        }
    };

    const handleSetupSubmit = async ({
                                         age,
                                         gender,
                                         ethnicity,
                                         image,
                                     }: PatientSetupSubmitPayload) => {
        setAge(age);
        setGender(gender);
        setEthnicity(ethnicity);

        setIsUploading(true);
        setShowProcessing(true);
        setShowNoFaceModal(false);
        setStatusText("Extracting face mesh...");
        setResults([]);
        setSelectedHPO(null);

        pendingSetupImageRef.current = image;
        pendingFaceDetectionRef.current = true;
        imageRef.current = image;

        try {
            await faceMeshDetectorRef.current?.send({image});
        } finally {
            setIsUploading(false);
        }
    };

    const reset = () => {
        setShowSetupFlow(true);
        setShowProcessing(false);
        setShowMainContent(false);
        setShowNoFaceModal(false);
        setIsPredicting(false);
        setIsUploading(false);

        setResults([]);
        setStatusText("");
        setSearchTerm("");
        setExpandedId(null);
        setSelectedHPO(null);
        setMinConfidence(0);

        setAge(0);
        setGender(-1);
        setEthnicity(-1);

        imageRef.current = null;
        pendingSetupImageRef.current = null;
        pendingFaceDetectionRef.current = false;
        landmarksRef.current = [];

        if (faceCanvasRef.current) {
            const ctx = faceCanvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, faceCanvasRef.current.width, faceCanvasRef.current.height);
        }
    };

    return (
        <>
            <Script
                src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
                strategy="afterInteractive"
            />
            <Script
                src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js"
                strategy="afterInteractive"
            />
            <Script
                src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"
                strategy="afterInteractive"
            />
            <Script
                src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"
                strategy="afterInteractive"
            />
            <Script
                src="https://cdn.plot.ly/plotly-3.4.0.min.js"
                strategy="afterInteractive"
            />
            <Script
                src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"
                strategy="afterInteractive"
                onLoad={initModels}
            />

            <style jsx global>{`
                .main-content {
                    min-height: 100vh;
                    height: 100vh;
                    max-height: 100vh;
                    padding: 2rem 0;
                    position: relative;
                    z-index: 1;
                }

                .face-display {
                    position: relative;
                    //background: rgba(255, 255, 255, 0.95);
                    //backdrop-filter: blur(10px);
                    //border-radius: 20px;
                    overflow: hidden;
                    //box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }

                .face-canvas {
                    width: auto;
                    height: 87vh;
                    display: block;
                    border: 1px solid rgba(0, 0, 0, 0.3);
                }

                .hpo-panel {
                    background: rgba(255, 255, 255, 0.95);
                    //backdrop-filter: blur(10px);
                    //border-radius: 20px;
                    height: 80vh;
                    overflow-y: auto;
                    //box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                }

                @media (max-width: 1024px) {
                    .hpo-panel {
                        overflow-y: hidden;
                        height: 100%;
                    }
                }

                .reset-btn {
                    position: fixed;
                    bottom: 2rem;
                    right: 2rem;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: #dc3545;
                    color: white;
                    border: none;
                    box-shadow: 0 10px 30px rgba(220, 53, 69, 0.3);
                    z-index: 1000;
                    transition: all 0.3s ease;
                }

                .reset-btn:hover {
                    transform: scale(1.1);
                    box-shadow: 0 15px 40px rgba(220, 53, 69, 0.4);
                }

                .search-box {
                    background: rgba(255, 255, 255, 0.95);
                    border: 1px solid rgba(0, 0, 0, 0.3);
                    border-radius: 50px;
                    padding: 1rem 1.5rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    margin-bottom: 1rem;
                }

                .loading-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(255, 255, 255, 0.96);
                    backdrop-filter: blur(10px);
                    z-index: 2000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }

                .status-chip {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.55rem 0.9rem;
                    border-radius: 999px;
                    background: rgba(13, 110, 253, 0.75);
                    color: #ffffff;
                    font-weight: 600;
                    font-size: 0.9rem;
                }

                .no-face-overlay {
                    position: fixed;
                    inset: 0;
                    z-index: 1200;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 1rem;
                    background: rgba(255, 255, 255, 0.75);
                    backdrop-filter: blur(6px);
                }

                .no-face-modal {
                    width: min(520px, 100%);
                    background: white;
                    border-radius: 24px;
                    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.16);
                    padding: 1.5rem;
                    border: 1px solid rgba(0, 0, 0, 0.06);
                }

                .legend-toggle {
                    border-radius: 999px;
                }

                .face-canvas-host {
                    //position: fixed;
                    //inset: 0;
                    //z-index: 0;
                    //display: flex;
                    //justify-content: center;
                    //align-items: center;
                    //pointer-events: none;
                }

                .face-canvas-host.hidden {
                    visibility: hidden;
                    opacity: 0;
                }

                .face-canvas-host.visible {
                    visibility: visible;
                    opacity: 1;
                }

                .face-canvas {
                    width: auto;
                    height: 87vh;
                    display: block;
                }
            `}</style>

            <AppInitializationProgress
                isVisible={showAppInitialization}
                modelStatus={modelStatus}
                loadingProgress={loadingProgress}
                loadingETA={loadingETA}
            />

            <PatientSetupFlow
                initialAge={age}
                initialGender={gender}
                initialEthnicity={ethnicity}
                isOpen={showSetupFlow}
                isSubmitting={isUploading || showProcessing}
                onSubmit={handleSetupSubmit}
            />

            {showNoFaceModal && (
                <div className="no-face-overlay">
                    <div className="no-face-modal">
                        <div className="d-flex align-items-start gap-3">
                            <div
                                className="d-inline-flex align-items-center justify-content-center rounded-circle bg-danger-subtle text-danger"
                                style={{width: 48, height: 48, flexShrink: 0}}
                            >
                                <i className="fas fa-face-frown"></i>
                            </div>

                            <div className="flex-grow-1">
                                <h4 className="mb-2">No face mesh detected</h4>
                                <p className="text-muted mb-3">
                                    Please upload a clearer frontal face image with good lighting and the full
                                    face visible.
                                </p>

                                <div className="d-flex gap-2 flex-wrap">
                                    <button
                                        type="button"
                                        className="btn btn-primary rounded-pill px-4"
                                        onClick={() => setShowNoFaceModal(false)}
                                    >
                                        Try another image
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className={`container-fluid main-content face-canvas-host ${showMainContent ? "visible" : "hidden"}`}>
                <div className="row g-4 h-100">
                    <div className="col-lg-6">
                        <div className="face-display h-100 p-3">
                            <div className="position-absolute top-0 start-0 m-3 d-flex gap-2 flex-wrap">
                                {statusText && <div className="status-chip">{statusText}</div>}
                                {isPredicting && (
                                    <div className="status-chip">
                        <span
                            className="spinner-border spinner-border-sm"
                            role="status"
                            aria-hidden="true"
                        ></span>
                                        Running predictions...
                                    </div>
                                )}
                            </div>

                            <canvas ref={faceCanvasRef} className="face-canvas"/>

                            <div className="position-absolute bottom-0 start-0 m-3 d-flex gap-2">
                                <button
                                    type="button"
                                    className="btn btn-outline-secondary legend-toggle"
                                    onClick={() => setShowColorLegend((prev) => !prev)}
                                >
                                    <i className="fas fa-palette me-2"></i>
                                    {showColorLegend ? "Hide legend" : "Show legend"}
                                </button>
                            </div>

                            {showColorLegend && (
                                <div
                                    className="position-absolute bottom-0 end-0 m-3 p-3 bg-white rounded-4 shadow-sm"
                                    style={{minWidth: 220}}
                                >
                                    <div className="fw-semibold mb-2">Importance legend</div>
                                    <div
                                        style={{
                                            height: 12,
                                            borderRadius: 999,
                                            background:
                                                "linear-gradient(90deg, #3939f6 0%, #bda87b 5%, #dd0303 100%)",
                                        }}
                                    />
                                    <div className="d-flex justify-content-between mt-2 small text-muted">
                                        <span>Low</span>
                                        <span>Threshold</span>
                                        <span>High</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="col-lg-6">
                        <div className="d-flex flex-column h-100">
                            <input
                                type="text"
                                className="search-box"
                                placeholder="Search HPO terms or IDs..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />

                            <div className="d-flex justify-content-between align-items-center mb-3 px-1">
                                <div className="text-muted small">
                                    {filterStats.visibleCount} results
                                    {searchTerm && ` • ${filterStats.matchCount} matches`}
                                </div>

                                <div className="d-flex align-items-center gap-2">
                                    <label className="small text-muted">Min confidence</label>
                                    <input
                                        type="range"
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        value={minConfidence}
                                        onChange={(e) => setMinConfidence(Number(e.target.value))}
                                    />
                                    <span className="small text-muted">
                        {Math.round(minConfidence * 100)}%
                      </span>
                                </div>
                            </div>

                            <div className="hpo-panel p-3 flex-grow-1">
                                <HPOTree
                                    nodes={hierarchy}
                                    searchTerm={searchTerm}
                                    minConfidence={minConfidence}
                                    expandedId={expandedId}
                                    onSelect={handleHPONodeSelect}
                                    onFilterStatsChange={setFilterStats}
                                    onToggleExpand={(id) => {
                                        setExpandedId((prev) => (prev === id ? null : id));
                                    }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <button className="reset-btn" onClick={reset} title="Start over">
                    <i className="fas fa-redo"></i>
                </button>
            </div>
        </>
    );
}