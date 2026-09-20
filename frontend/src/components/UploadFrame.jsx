import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadFrame({ file, previewUrl, status, resultTone, onFileSelected }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (fileList) => {
      const picked = fileList?.[0];
      if (picked) onFileSelected(picked);
    },
    [onFileSelected]
  );

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFiles(event.dataTransfer.files);
  };

  const toneClass = status === "done" ? `frame--${resultTone}` : "";

  return (
    <div
      className={`upload-frame ${isDragging ? "upload-frame--dragging" : ""} ${toneClass}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <span className="upload-frame__corner upload-frame__corner--tl" />
      <span className="upload-frame__corner upload-frame__corner--tr" />
      <span className="upload-frame__corner upload-frame__corner--bl" />
      <span className="upload-frame__corner upload-frame__corner--br" />

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        onChange={(event) => handleFiles(event.target.files)}
        hidden
      />

      {previewUrl ? (
        <div className="upload-frame__preview">
          <img src={previewUrl} alt="Uploaded MRI scan" />
          {status === "analyzing" && <div className="scan-sweep" aria-hidden="true" />}
        </div>
      ) : (
        <button
          type="button"
          className="upload-frame__placeholder"
          onClick={() => inputRef.current?.click()}
        >
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <path
              d="M20 26V10M20 10l-6 6M20 10l6 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M8 26v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
          <p>Drop an MRI scan here, or click to browse</p>
        </button>
      )}

      {file && (
        <p className="upload-frame__meta">
          {file.name} — {formatBytes(file.size)}
        </p>
      )}

      {!file && <p className="upload-frame__meta">jpg, png, bmp or webp, up to 10 MB</p>}

      {file && status !== "analyzing" && (
        <button
          type="button"
          className="upload-frame__change"
          onClick={() => inputRef.current?.click()}
        >
          Choose a different scan
        </button>
      )}
    </div>
  );
}
