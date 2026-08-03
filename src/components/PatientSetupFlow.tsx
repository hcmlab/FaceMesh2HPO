"use client";

import React, {ChangeEvent, DragEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState,} from "react";

export type PatientSetupSubmitPayload = {
    age: number;
    gender: number;
    ethnicity: number;
    image: HTMLImageElement;
};

interface PatientSetupFlowProps {
    initialAge?: number;
    initialGender?: number;
    initialEthnicity?: number;
    isOpen: boolean;
    isSubmitting?: boolean;
    onSubmit: (payload: PatientSetupSubmitPayload) => Promise<void> | void;
}

const DEFAULTS = {
    age: 0,
    gender: -1,
    ethnicity: -1,
};

export const PatientSetupFlow = ({
                                     initialAge = DEFAULTS.age,
                                     initialGender = DEFAULTS.gender,
                                     initialEthnicity = DEFAULTS.ethnicity,
                                     isOpen,
                                     isSubmitting = false,
                                     onSubmit,
                                 }: PatientSetupFlowProps) => {
    const [localAge, setLocalAge] = useState(initialAge);
    const [localGender, setLocalGender] = useState(initialGender);
    const [localEthnicity, setLocalEthnicity] = useState(initialEthnicity);

    const [dragActive, setDragActive] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [selectedImage, setSelectedImage] = useState<HTMLImageElement | null>(null);
    const [fileName, setFileName] = useState("");
    const [errorText, setErrorText] = useState("");

    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!isOpen) return;

        setLocalAge(initialAge);
        setLocalGender(initialGender);
        setLocalEthnicity(initialEthnicity);
    }, [initialAge, initialGender, initialEthnicity, isOpen]);

    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    const isDefault =
        localAge === DEFAULTS.age &&
        localGender === DEFAULTS.gender &&
        localEthnicity === DEFAULTS.ethnicity;

    const canSubmit = useMemo(() => {
        return !!selectedImage && !isSubmitting;
    }, [selectedImage, isSubmitting]);

    const resetImageSelection = useCallback(() => {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
        setSelectedImage(null);
        setFileName("");
        setDragActive(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }, [previewUrl]);

    const handleDrag = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();

        if (isSubmitting) return;

        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, [isSubmitting]);

    const createImageFromFile = useCallback((file: File) => {
        const objectUrl = URL.createObjectURL(file);
        const img = new Image();

        img.onload = () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
            setPreviewUrl(objectUrl);
            setSelectedImage(img);
            setFileName(file.name);
            setErrorText("");
        };

        img.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            setErrorText("The selected file could not be loaded as an image.");
            setSelectedImage(null);
            setFileName("");
        };

        img.src = objectUrl;
    }, [previewUrl]);

    const handleFile = useCallback((file: File) => {
        if (!file.type.startsWith("image/")) {
            setErrorText("Please select an image file (JPG, PNG, WebP).");
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            setErrorText("File too large. Maximum size is 5MB.");
            return;
        }

        createImageFromFile(file);
    }, [createImageFromFile]);

    const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (isSubmitting) return;

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, [handleFile, isSubmitting]);

    const openFileDialog = useCallback(() => {
        if (isSubmitting) return;
        fileInputRef.current?.click();
    }, [isSubmitting]);

    const handleFileInput = useCallback((e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    }, [handleFile]);

    const handleUploadAreaClick = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        openFileDialog();
    }, [openFileDialog]);

    const handleUploadAreaKeyDown = useCallback((e: KeyboardEvent<HTMLDivElement>) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFileDialog();
        }
    }, [openFileDialog]);

    const handleSubmit = useCallback(async () => {
        if (!selectedImage) {
            setErrorText("Please upload a face image before submitting.");
            return;
        }

        setErrorText("");

        await onSubmit({
            age: localAge,
            gender: localGender,
            ethnicity: localEthnicity,
            image: selectedImage,
        });
    }, [selectedImage, onSubmit, localAge, localGender, localEthnicity]);

    if (!isOpen) return null;

    return (
        <>
            <style jsx>{`
                .setup-overlay {
                    position: fixed;
                    inset: 0;
                    width: 100vw;
                    height: 100vh;
                    z-index: 1050;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 1.5rem;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                }

                .setup-shell {
                    width: min(980px, 100%);
                    max-height: calc(100vh - 3rem);
                    overflow-y: auto;
                    border-radius: 28px;
                    background: rgba(255, 255, 255, 0.98);
                    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.6);
                }

                .setup-header {
                    padding: 1.75rem 1.75rem 1rem;
                    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
                }

                .setup-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.45rem 0.8rem;
                    border-radius: 999px;
                    background: rgba(102, 126, 234, 0.12);
                    color: #667eea;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 0.9rem;
                }

                .setup-title {
                    margin: 0 0 0.4rem;
                    color: #212529;
                    font-size: 1.9rem;
                    font-weight: 700;
                    line-height: 1.15;
                }

                .setup-subtitle {
                    margin: 0;
                    color: #6c757d;
                    font-size: 1rem;
                }

                .setup-body {
                    padding: 1.75rem;
                }

                .setup-grid {
                    display: grid;
                    grid-template-columns: minmax(320px, 1fr) minmax(320px, 1.1fr);
                    gap: 1.25rem;
                }

                .setup-panel {
                    background: linear-gradient(180deg, rgba(248, 249, 250, 0.92), rgba(255, 255, 255, 0.92));
                    border: 1px solid rgba(0, 0, 0, 0.06);
                    border-radius: 22px;
                    padding: 1.25rem;
                    height: 100%;
                }

                .panel-title {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                }

                .panel-icon {
                    width: 42px;
                    height: 42px;
                    border-radius: 14px;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1rem;
                    box-shadow: 0 10px 24px rgba(102, 126, 234, 0.25);
                }

                .panel-title-text {
                    display: flex;
                    flex-direction: column;
                }

                .panel-title-text strong {
                    color: #212529;
                    font-size: 1rem;
                    line-height: 1.2;
                }

                .panel-title-text span {
                    color: #6c757d;
                    font-size: 0.9rem;
                }

                .form-label-custom {
                    display: block;
                    font-size: 0.92rem;
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 0.45rem;
                }

                .help-text {
                    margin-top: 0.35rem;
                    font-size: 0.8rem;
                    color: #8b949e;
                }

                .form-control,
                .form-select {
                    border-radius: 14px;
                    min-height: 48px;
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    box-shadow: none !important;
                }

                .form-control:focus,
                .form-select:focus {
                    border-color: #667eea;
                    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.15) !important;
                }

                .upload-zone {
                    position: relative;
                    border: 2px dashed ${dragActive ? "#667eea" : "rgba(102, 126, 234, 0.25)"};
                    background: ${dragActive ? "rgba(102, 126, 234, 0.08)" : "rgba(255,255,255,0.85)"};
                    border-radius: 20px;
                    min-height: 320px;
                    padding: 1.25rem;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    transition: all 0.25s ease;
                    cursor: ${isSubmitting ? "not-allowed" : "pointer"};
                    opacity: ${isSubmitting ? 0.7 : 1};
                }

                .upload-zone:hover {
                    border-color: #667eea;
                    transform: ${isSubmitting ? "none" : "translateY(-1px)"};
                }

                .upload-icon {
                    font-size: 3rem;
                    color: #667eea;
                    margin-bottom: 0.85rem;
                }

                .upload-title {
                    color: #343a40;
                    font-weight: 700;
                    font-size: 1.05rem;
                    margin-bottom: 0.35rem;
                }

                .upload-subtitle {
                    color: #6c757d;
                    font-size: 0.92rem;
                    margin-bottom: 1rem;
                    max-width: 320px;
                }

                .upload-meta {
                    color: #8b949e;
                    font-size: 0.82rem;
                    margin-top: 0.75rem;
                }

                .choose-btn {
                    border-radius: 999px;
                    padding: 0.65rem 1.05rem;
                    font-weight: 600;
                }

                .preview-shell {
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.85rem;
                }

                .preview-image {
                    display: block;
                    max-width: 100%;
                    max-height: 220px;
                    object-fit: contain;
                    border-radius: 18px;
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.12);
                    background: white;
                }

                .preview-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.55rem 0.9rem;
                    border-radius: 999px;
                    background: rgba(25, 135, 84, 0.12);
                    color: #198754;
                    font-weight: 600;
                    font-size: 0.88rem;
                }

                .filename {
                    color: #495057;
                    font-size: 0.88rem;
                    word-break: break-word;
                    max-width: 100%;
                }

                .replace-row {
                    display: flex;
                    gap: 0.75rem;
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .processing-box {
                    margin-top: 1rem;
                    border-radius: 16px;
                    padding: 1rem;
                    background: rgba(102, 126, 234, 0.08);
                    border: 1px solid rgba(102, 126, 234, 0.12);
                    color: #4c63d2;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.75rem;
                    font-weight: 600;
                }

                .error-box {
                    margin-top: 1rem;
                    border-radius: 16px;
                    padding: 0.95rem 1rem;
                    background: rgba(220, 53, 69, 0.08);
                    border: 1px solid rgba(220, 53, 69, 0.15);
                    color: #b02a37;
                    font-size: 0.92rem;
                }

                .setup-footer {
                    padding: 1rem 1.75rem 1.75rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                }

                .footer-note {
                    color: #6c757d;
                    font-size: 0.9rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .footer-actions {
                    display: flex;
                    gap: 0.75rem;
                    flex-wrap: wrap;
                }

                .skip-btn,
                .submit-btn {
                    border-radius: 999px;
                    padding: 0.75rem 1.2rem;
                    font-weight: 600;
                    min-width: 160px;
                }

                .submit-btn {
                    box-shadow: 0 12px 28px rgba(102, 126, 234, 0.24);
                }

                .submit-btn:disabled {
                    box-shadow: none;
                }

                @media (max-width: 900px) {
                    .setup-grid {
                        grid-template-columns: 1fr;
                    }
                }

                @media (max-width: 640px) {
                    .setup-overlay {
                        padding: 0.75rem;
                    }

                    .setup-shell {
                        max-height: calc(100vh - 1.5rem);
                        border-radius: 22px;
                    }

                    .setup-header,
                    .setup-body,
                    .setup-footer {
                        padding-left: 1rem;
                        padding-right: 1rem;
                    }

                    .setup-title {
                        font-size: 1.55rem;
                    }

                    .footer-actions {
                        width: 100%;
                    }

                    .skip-btn,
                    .submit-btn {
                        width: 100%;
                    }
                }
            `}</style>

            <div className="setup-overlay" aria-modal="true" role="dialog">
                <div className="setup-shell">
                    <div className="setup-header">
                        <h2 className="setup-title">Patient Setup</h2>
                        <p className="setup-subtitle">
                            Enter demographics and upload a face image in one step, then start inference with a single
                            submit action.
                        </p>
                    </div>

                    <div className="setup-body">
                        <div className="setup-grid">
                            <div className="setup-panel">
                                <div className="panel-title">
                                    <div className="panel-icon">
                                        <i className="fas fa-user"></i>
                                    </div>
                                    <div className="panel-title-text">
                                        <strong>Demographics</strong>
                                        <span>Optional, but can improve model inputs</span>
                                    </div>
                                </div>

                                <div className="row g-3">
                                    <div className="col-12">
                                        <label className="form-label-custom" htmlFor="patient-age">
                                            Age
                                        </label>
                                        <input
                                            id="patient-age"
                                            type="number"
                                            min={0}
                                            className="form-control"
                                            value={localAge}
                                            onChange={(e) => setLocalAge(Number(e.target.value))}
                                            disabled={isSubmitting}
                                        />
                                        <div className="help-text">
                                            Use 0 if age is unknown.
                                        </div>
                                    </div>

                                    <div className="col-12">
                                        <label className="form-label-custom" htmlFor="patient-gender">
                                            Sex
                                        </label>
                                        <select
                                            id="patient-gender"
                                            className="form-select"
                                            value={localGender}
                                            onChange={(e) => setLocalGender(Number(e.target.value))}
                                            disabled={isSubmitting}
                                        >
                                            <option value={-1}>Unknown</option>
                                            <option value={0}>Female</option>
                                            <option value={1}>Male</option>
                                            <option value={2}>Other</option>
                                        </select>
                                    </div>

                                    <div className="col-12">
                                        <label className="form-label-custom" htmlFor="patient-ethnicity">
                                            Ethnicity
                                        </label>
                                        <select
                                            id="patient-ethnicity"
                                            className="form-select"
                                            value={localEthnicity}
                                            onChange={(e) => setLocalEthnicity(Number(e.target.value))}
                                            disabled={isSubmitting}
                                        >
                                            <option value={-1}>Unknown</option>
                                            <option value={0}>African</option>
                                            <option value={1}>Asian</option>
                                            <option value={2}>European</option>
                                            <option value={3}>Other</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <div className="setup-panel">
                                <div className="panel-title">
                                    <div className="panel-icon">
                                        <i className="fas fa-image"></i>
                                    </div>
                                    <div className="panel-title-text">
                                        <strong>Face image</strong>
                                        <span>Drag and drop or browse a frontal image</span>
                                    </div>
                                </div>

                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    className="d-none"
                                    onChange={handleFileInput}
                                    disabled={isSubmitting}
                                />

                                <div
                                    className="upload-zone"
                                    onClick={handleUploadAreaClick}
                                    onKeyDown={handleUploadAreaKeyDown}
                                    onDragEnter={handleDrag}
                                    onDragLeave={handleDrag}
                                    onDragOver={handleDrag}
                                    onDrop={handleDrop}
                                    tabIndex={0}
                                    role="button"
                                    aria-label="Upload face image"
                                >
                                    {!previewUrl && (
                                        <>
                                            <div className="upload-icon">
                                                <i className="fas fa-cloud-upload-alt"></i>
                                            </div>
                                            <div className="upload-title">
                                                {dragActive ? "Drop image here" : "Drop your face image here"}
                                            </div>
                                            <div className="upload-subtitle">
                                                Or click to browse and select a single-face image.
                                            </div>
                                            <button
                                                type="button"
                                                className="btn btn-outline-primary choose-btn"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    openFileDialog();
                                                }}
                                                disabled={isSubmitting}
                                            >
                                                <i className="fas fa-folder-open me-2"></i>
                                                Choose file
                                            </button>
                                            <div className="upload-meta">
                                                JPG, PNG, WebP • Max 5MB • Single face preferred
                                            </div>
                                        </>
                                    )}

                                    {previewUrl && (
                                        <div className="preview-shell">
                                            <div className="preview-pill">
                                                <i className="fas fa-check-circle"></i>
                                                Ready for submission
                                            </div>

                                            <img
                                                src={previewUrl}
                                                alt="Selected face preview"
                                                className="preview-image"
                                            />

                                            <div className="filename">
                                                <i className="fas fa-file-image me-2 text-secondary"></i>
                                                {fileName}
                                            </div>

                                            <div className="replace-row">
                                                <button
                                                    type="button"
                                                    className="btn btn-outline-primary choose-btn"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        openFileDialog();
                                                    }}
                                                    disabled={isSubmitting}
                                                >
                                                    <i className="fas fa-arrows-rotate me-2"></i>
                                                    Replace image
                                                </button>

                                                <button
                                                    type="button"
                                                    className="btn btn-outline-secondary choose-btn"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        resetImageSelection();
                                                    }}
                                                    disabled={isSubmitting}
                                                >
                                                    <i className="fas fa-trash me-2"></i>
                                                    Remove
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {isSubmitting && (
                                    <div className="processing-box">
                    <span
                        className="spinner-border spinner-border-sm"
                        role="status"
                        aria-hidden="true"
                    />
                                        Processing image and extracting face mesh...
                                    </div>
                                )}

                                {errorText && <div className="error-box">{errorText}</div>}
                            </div>
                        </div>
                    </div>

                    <div className="setup-footer">
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: "0.5rem",
                                alignItems: "flex-start",
                            }}
                        >
                            <div className="footer-note">
                                <i className="fas fa-circle-info text-primary"></i>
                                {isDefault
                                    ? "You can submit with default demographics after selecting an image."
                                    : "Demographic metadata has been customized for this submission."}
                            </div>

                            <div
                                style={{
                                    color: "#6c757d",
                                    fontSize: "0.82rem",
                                    lineHeight: 1.5,
                                }}
                            >
                                <div>
                                    © 2026 Human-Centered Artificial Intelligence Lab, University of Augsburg,
                                    Germany
                                </div>
                                <div>
                                    Publication:{" "}
                                    <a
                                        href="https://doi.org/10.48550/arXiv.2607.05585"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        DOI
                                    </a>
                                    {" · "}
                                    GitHub:{" "}
                                    <a
                                        href="https://github.com/hcmlab/FaceMesh2HPO"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        FaceMesh2HPO
                                    </a>
                                    {" · "}
                                    Issue Reporting:{" "}
                                    <a
                                        href="https://github.com/hcmlab/FaceMesh2HPO/issues"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        GitHub Issues
                                    </a>
                                </div>
                            </div>
                        </div>

                        <div className="footer-actions">
                            <button
                                type="button"
                                className="btn btn-light skip-btn"
                                disabled={isSubmitting}
                                onClick={() => {
                                    setLocalAge(DEFAULTS.age);
                                    setLocalGender(DEFAULTS.gender);
                                    setLocalEthnicity(DEFAULTS.ethnicity);
                                }}
                            >
                                <i className="fas fa-rotate-left me-2"></i>
                                Reset demographics
                            </button>

                            <button
                                type="button"
                                className="btn btn-primary submit-btn"
                                disabled={!canSubmit}
                                onClick={handleSubmit}
                            >
                                {isSubmitting ? (
                                    <>
                    <span
                        className="spinner-border spinner-border-sm me-2"
                        role="status"
                        aria-hidden="true"
                    />
                                        Starting...
                                    </>
                                ) : (
                                    <>
                                        <i className="fas fa-play me-2"></i>
                                        Submit
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default PatientSetupFlow;