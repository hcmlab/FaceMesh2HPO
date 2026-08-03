import React from "react";
import Image from "next/image";
import favicon from "@/app/android-chrome-192x192.png";

interface AppInitializationProgressProps {
    isVisible: boolean;
    modelStatus: string;
    loadingProgress: number;
    loadingETA: string | null;
}

export const AppInitializationProgress = ({
    isVisible,
    modelStatus,
    loadingProgress,
    loadingETA,
}: AppInitializationProgressProps) => {
    if (!isVisible) return null;

    return (
        <>
            <style jsx global>{`
                .loading-overlay {
                    position: fixed;
                    inset: 0;
                    z-index: 2000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 1rem;
                    background: rgba(255, 255, 255, 0.96);
                    backdrop-filter: blur(10px);
                    overflow-y: auto;
                    -webkit-overflow-scrolling: touch;
                }

                .app-init-shell {
                    width: min(960px, calc(100vw - 2rem));
                    max-height: calc(100dvh - 2rem);
                    padding: 2rem;
                    border-radius: 28px;
                    background: rgba(255, 255, 255, 0.9);
                    backdrop-filter: blur(18px);
                    border: 1px solid rgba(255, 255, 255, 0.7);
                    box-shadow:
                        0 20px 60px rgba(15, 23, 42, 0.12),
                        0 8px 24px rgba(15, 23, 42, 0.08);
                    overflow-y: auto;
                    overscroll-behavior: contain;
                }

                .app-init-badge {
                    display: inline-flex;
                    align-items: center;
                    padding: 0.45rem 0.85rem;
                    border-radius: 999px;
                    background: rgba(13, 110, 253, 0.08);
                    color: #0d6efd;
                    font-size: 0.85rem;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                    margin-bottom: 1rem;
                }

                .app-init-header {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                }

                .app-init-logo {
                    width: 64px;
                    height: 64px;
                    border-radius: 20px;
                    display: block;
                    object-fit: contain;
                    box-shadow: 0 12px 24px rgba(13, 110, 253, 0.12);
                    flex-shrink: 0;
                }

                .app-init-title {
                    margin: 0;
                    font-size: clamp(1.8rem, 4vw, 3.2rem);
                    line-height: 1.05;
                    font-weight: 800;
                    letter-spacing: -0.03em;
                    color: #0f172a;
                    word-break: break-word;
                }

                .app-init-subtitle {
                    margin: 0.5rem 0 0;
                    max-width: 60ch;
                    color: #475569;
                    font-size: 1.02rem;
                    line-height: 1.6;
                }

                .app-init-card {
                    background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.95));
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 22px;
                    padding: 1.25rem;
                    margin-bottom: 1.5rem;
                }

                .app-init-progress {
                    height: 12px;
                    border-radius: 999px;
                    overflow: hidden;
                    background: rgba(148, 163, 184, 0.18);
                }

                .app-init-progress .progress-bar {
                    border-radius: 999px;
                    background: linear-gradient(90deg, #0d6efd, #4dabf7);
                }

                .app-init-progress-meta {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 0.75rem;
                    margin-top: 0.5rem;
                    flex-wrap: wrap;
                }

                .app-init-progress-label {
                    font-size: 0.9rem;
                    font-weight: 600;
                }

                .app-init-info-grid {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 1rem;
                }

                .app-init-info-card {
                    padding: 1.1rem 1.15rem;
                    border-radius: 20px;
                    background: rgba(255, 255, 255, 0.72);
                    border: 1px solid rgba(148, 163, 184, 0.14);
                }

                .app-init-info-title {
                    font-weight: 700;
                    color: #0f172a;
                    margin-bottom: 0.45rem;
                }

                .app-init-info-text {
                    margin: 0;
                    color: #526075;
                    line-height: 1.65;
                    font-size: 0.96rem;
                }

                .app-init-footer {
                    margin-top: 1.25rem;
                    color: #64748b;
                    font-size: 0.92rem;
                    text-align: center;
                }

                @media (max-width: 768px) {
                    .loading-overlay {
                        align-items: flex-start;
                        padding: 0.75rem;
                    }

                    .app-init-shell {
                        width: 100%;
                        max-height: calc(100svh - 1.5rem);
                        padding: 1.25rem;
                        border-radius: 22px;
                    }

                    .app-init-header {
                        align-items: flex-start;
                        gap: 0.875rem;
                    }

                    .app-init-info-grid {
                        grid-template-columns: 1fr;
                    }

                    .app-init-logo {
                        width: 54px;
                        height: 54px;
                        border-radius: 16px;
                    }

                    .app-init-subtitle {
                        font-size: 0.95rem;
                        line-height: 1.5;
                    }

                    .app-init-card {
                        padding: 1rem;
                        border-radius: 18px;
                    }
                }

                @media (max-width: 520px) {
                    .loading-overlay {
                        padding: 0.5rem;
                    }

                    .app-init-shell {
                        width: 100%;
                        max-height: calc(100svh - 1rem);
                        padding: 1rem;
                        border-radius: 18px;
                    }

                    .app-init-badge {
                        font-size: 0.78rem;
                        padding: 0.4rem 0.72rem;
                        margin-bottom: 0.85rem;
                    }

                    .app-init-header {
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 0.75rem;
                        margin-bottom: 1rem;
                    }

                    .app-init-logo {
                        width: 48px;
                        height: 48px;
                        border-radius: 14px;
                    }

                    .app-init-title {
                        font-size: clamp(1.5rem, 8vw, 2rem);
                        line-height: 1.1;
                    }

                    .app-init-subtitle {
                        margin-top: 0.35rem;
                        font-size: 0.9rem;
                    }

                    .app-init-card {
                        padding: 0.9rem;
                        margin-bottom: 1rem;
                    }

                    .app-init-info-card {
                        padding: 0.9rem;
                        border-radius: 16px;
                    }

                    .app-init-info-title {
                        font-size: 0.95rem;
                    }

                    .app-init-info-text,
                    .app-init-footer,
                    .app-init-progress-label {
                        font-size: 0.85rem;
                    }
                }
            `}</style>
            <div className="loading-overlay">
                <div className="app-init-shell">
                    <div className="app-init-badge">Prototype for research purposes only!</div>

                    <div className="app-init-header">
                        <Image
                            src={favicon}
                            alt="FaceMesh2HPO logo"
                            width={64}
                            height={64}
                            className="app-init-logo"
                            priority
                        />
                        <div>
                            <h1 className="app-init-title">FaceMesh2HPO</h1>
                            <p className="app-init-subtitle">
                                Automated facial phenotype screening from 2D images to Human Phenotype Ontology terms.
                            </p>
                        </div>
                    </div>

                    <div className="app-init-card">
                        <div className="d-flex align-items-center gap-3 mb-3">
                            <div className="spinner-border text-primary" role="status" />
                            <div>
                                <h4 className="mb-1">{modelStatus}</h4>
                                <p className="text-muted mb-0">
                                    Preparing the face-mesh pipeline and HPO classification models.
                                </p>
                            </div>
                        </div>

                        <div className="progress app-init-progress">
                            <div
                                className="progress-bar progress-bar-striped progress-bar-animated"
                                style={{ width: `${loadingProgress}%` }}
                            />
                        </div>

                        <div className="app-init-progress-meta">
                            <span className="app-init-progress-label">
                                {Math.round(loadingProgress)}% loaded
                            </span>
                            {loadingETA && (
                                <span className="app-init-progress-label text-muted">{loadingETA}</span>
                            )}
                        </div>
                    </div>

                    <div className="app-init-info-grid">
                        <div className="app-init-info-card">
                            <div className="app-init-info-title">What the app does</div>
                            <p className="app-init-info-text">
                                FaceMesh2HPO helps classify craniofacial phenotype descriptors as HPO terms from 2D
                                facial images, supporting more structured phenotypic assessment workflows.
                            </p>
                        </div>

                        <div className="app-init-info-card">
                            <div className="app-init-info-title">How it works</div>
                            <p className="app-init-info-text">
                                The pipeline detects dense facial mesh landmarks, compares geometric patterns, and
                                applies PointNet-based classifiers to infer phenotype terms across multiple levels of
                                specificity.
                            </p>
                        </div>

                        <div className="app-init-info-card">
                            <div className="app-init-info-title">Clinical context</div>
                            <p className="app-init-info-text">
                                HPO provides a standardized vocabulary for phenotypic abnormalities, which supports
                                consistent clinical description and phenotype-based analysis.
                            </p>
                        </div>

                        <div className="app-init-info-card">
                            <div className="app-init-info-title">Why this matters</div>
                            <p className="app-init-info-text">
                                The approach is designed to reduce manual overhead for medical experts and help
                                streamline patient phenotyping, especially when expert facial pattern recognition is
                                difficult to acquire and scale.
                            </p>
                        </div>
                    </div>

                    <div className="app-init-footer">
                        Built for structured facial phenotyping with face meshes, HPO labels, and model-assisted
                        decision support.
                    </div>
                </div>
            </div>
        </>
    );
};