import { useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import UploadFrame from "./components/UploadFrame";
import ResultPanel from "./components/ResultPanel";
import Footer from "./components/Footer";
import { predict, ApiError } from "./api";
import "./App.css";

const TONE_BY_CLASS = { no: "safe", yes: "alert" };

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | ready | analyzing | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const previewUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function handleFileSelected(nextFile) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const url = URL.createObjectURL(nextFile);
    previewUrlRef.current = url;

    setFile(nextFile);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
    setStatus("ready");
  }

  async function handleAnalyze() {
    setStatus("analyzing");
    setError(null);
    try {
      const response = await predict(file);
      setResult(response);
      setStatus("done");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
      setStatus("error");
    }
  }

  const tone = result ? (TONE_BY_CLASS[result.predicted_class] ?? "safe") : null;
  const canAnalyze = file && (status === "ready" || status === "error");

  return (
    <div className="app">
      <Header />

      <main className="app__main">
        <UploadFrame
          file={file}
          previewUrl={previewUrl}
          status={status}
          resultTone={tone}
          onFileSelected={handleFileSelected}
        />

        {canAnalyze && (
          <button type="button" className="analyze-button" onClick={handleAnalyze}>
            Analyze scan
          </button>
        )}

        {status === "analyzing" && <p className="status-line">Analyzing scan…</p>}

        <ResultPanel status={status} result={result} error={error} tone={tone} />
      </main>

      <Footer />
    </div>
  );
}
