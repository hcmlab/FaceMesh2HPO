import React, {ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState} from "react";

interface ImageUploadProps {
    onImageSelect: (img: HTMLImageElement) => void;
    isUploading?: boolean;
}

export const ImageUpload = ({
                                onImageSelect,
                                isUploading = false
                            }: ImageUploadProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    // Drag & Drop Events
    const handleDrag = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(e.type === "dragenter" || e.type === "dragover");
    }, []);

    const handleFile = useCallback((file: File) => {
        // Validierung
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, WebP)');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('File too large! Max 5MB.');
            return;
        }

        // Preview generieren
        const url = URL.createObjectURL(file);
        setPreviewUrl(url);

        const img = new Image();
        img.onload = async () => {
            onImageSelect(img);
        };
        img.src = URL.createObjectURL(file);
    }, [onImageSelect]);

    const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, []);

    const openFileDialog = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileInput = useCallback((e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    }, []);

    const handleOverlayClick = useCallback((e: React.MouseEvent) => {
        if (e.target === e.currentTarget) {
            openFileDialog();
        }
    }, [openFileDialog]);

    const handleUploadAreaClick = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        openFileDialog();
    }, [openFileDialog]);

    // Cleanup
    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    return (
        <>
            <style jsx global>{`
                input[type="file"].d-none {
                    position: fixed !important;
                    top: -100% !important;
                    left: -100% !important;
                    opacity: 0;
                    z-index: 9999;
                    width: 1px;
                    height: 1px;
                }

                .upload-area {
                    width: 500px;
                    max-width: 90vw;
                    height: 550px;
                    border: 4px dashed #667eea;
                    border-radius: 20px;
                    background: rgba(102, 126, 234, 0.1);
                    transition: all 0.3s ease;
                    cursor: pointer;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                    overflow: hidden;
                    padding: 2rem;
                    pointer-events: auto;
                }

                .upload-area:hover {
                    border-color: #764ba2;
                    background: rgba(102, 126, 234, 0.15);
                    transform: scale(1.02);
                }

                .upload-area.processing {
                    border-color: #28a745;
                    background: rgba(40, 167, 69, 0.1);
                    pointer-events: none;
                }
            `}</style>
            <div
                className={`file-upload-overlay position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center p-4
                  ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={handleOverlayClick}
            >
                {/* Hidden File Input */}
                <input
                    ref={fileInputRef}
                    className="d-none"
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handleFileInput}
                    tabIndex={-1}
                />

                <div
                    className={`upload-area p-5 rounded-4 shadow-lg text-center transition-all position-relative
                    ${isUploading ? 'uploading' : ''}`}
                    style={{
                        maxWidth: '450px',
                        width: '90vw',
                        cursor: isUploading ? 'wait' : 'pointer',
                        transition: 'all 0.3s ease'
                    }}
                    onClick={handleUploadAreaClick}
                >
                    {/* Loading Spinner */}
                    {isUploading && (
                        <div
                            className="position-absolute top-0 start-0 w-100 h-100 bg-white bg-opacity-75 rounded-4 d-flex align-items-center justify-content-center">
                            <div className="spinner-border spinner-border-sm text-primary" role="status">
                                <span className="visually-hidden">Processing...</span>
                            </div>
                            <div className="ms-2 small fw-bold text-primary">Analyzing face...</div>
                        </div>
                    )}

                    {/* Preview */}
                    {previewUrl && !isUploading && (
                        <div className="mb-4">
                            <img
                                src={previewUrl}
                                alt="Preview"
                                className="rounded-3 shadow-sm"
                                style={{
                                    width: '200px',
                                    height: '200px',
                                    objectFit: 'cover'
                                }}
                            />
                            <div className="small text-success mt-1">
                                <i className="fas fa-check-circle me-1"></i>
                                Ready!
                            </div>
                        </div>
                    )}

                    {/* Upload Area */}
                    {!previewUrl && (
                        <>
                            <i
                                className={`fas fa-cloud-upload-alt upload-icon mb-3 ${dragActive ? 'text-primary' : 'text-muted'}`}
                                style={{fontSize: '4rem'}}
                            ></i>

                            <div className="upload-text fw-bold mb-2">
                                {dragActive ? 'Drop image here!' : 'Drop your face image here'}
                            </div>

                            <div className="upload-subtext mb-4 text-muted">
                                or <span className="text-primary fw-bold cursor-pointer">click to browse</span><br/>
                                <small>JPG, PNG, WebP • Max 5MB • Single face preferred</small>
                            </div>
                        </>
                    )}

                    {/* Progress Bar */}
                    {isUploading && previewUrl && (
                        <div className="progress mb-3 mx-4" style={{height: '6px'}}>
                            <div
                                className="progress-bar progress-bar-striped progress-bar-animated bg-primary"
                                role="progressbar"
                                style={{width: '75%'}}
                            ></div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};
