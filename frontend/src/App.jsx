import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export default function App() {
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);

  const [mode, setMode] = useState("webcam"); // "webcam" | "upload"
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [name, setName] = useState("");

  const [statusMessage, setStatusMessage] = useState("Camera ready for scanning or enrollment.");
  const [statusType, setStatusType] = useState("neutral"); // "neutral" | "success" | "error"
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [loading, setLoading] = useState(false);

  // Start or stop webcam based on active mode
  useEffect(() => {
    if (mode !== "webcam") return;

    let activeStream = null;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatusType("neutral");
        setStatusMessage("Camera ready for scanning or enrollment.");
      } catch (err) {
        setStatusType("error");
        setStatusMessage("Unable to access webcam: " + err.message);
      }
    }

    startCamera();

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [mode]);

  const clearOverlay = () => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawBoundingBoxes = (faces, sourceWidth, sourceHeight, mirror = false) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      if (!face.box || face.box.length < 4) return;
      const [rawX, y, w, h] = face.box;
      const x = mirror ? sourceWidth - rawX - w : rawX;

      const color = face.recognized ? "#10b981" : "#ef4444";
      const label = face.recognized
        ? `${face.name}${typeof face.score === "number" ? ` (${face.score.toFixed(2)})` : ""}`
        : "Cannot recognize";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.font = "600 15px Inter, system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelY = y > 28 ? y - 28 : y + h;

      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, textWidth + 16, 28);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + 8, labelY + 19);
    });
  };

  const handleModeSwitch = (nextMode) => {
    setMode(nextMode);
    setDetectedFaces([]);
    clearOverlay();
    if (nextMode === "webcam") {
      setUploadedFile(null);
      setPreviewUrl(null);
    } else {
      setStatusType("neutral");
      setStatusMessage("Select a JPG or PNG photo to scan or enroll.");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setDetectedFaces([]);
    clearOverlay();
    setStatusType("neutral");
    setStatusMessage(`Loaded "${file.name}"`);
  };

  const getImagePayload = () =>
    new Promise((resolve, reject) => {
      if (mode === "upload") {
        if (!uploadedFile) {
          reject(new Error("Please choose an image file first."));
          return;
        }
        const img = new Image();
        img.onload = () =>
          resolve({
            blob: uploadedFile,
            filename: uploadedFile.name,
            width: img.naturalWidth,
            height: img.naturalHeight,
            mirrored: false,
          });
        img.onerror = () => reject(new Error("Invalid image file."));
        img.src = previewUrl;
        return;
      }

      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.videoWidth === 0) {
        reject(new Error("Webcam feed is not ready yet."));
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve({
              blob,
              filename: "webcam.jpg",
              width: video.videoWidth,
              height: video.videoHeight,
              mirrored: true,
            });
          } else {
            reject(new Error("Failed to capture webcam frame."));
          }
        },
        "image/jpeg",
        0.92
      );
    });

  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setStatusType("error");
      setStatusMessage("Please enter a person's name before enrolling.");
      return;
    }

    setLoading(true);
    setDetectedFaces([]);
    clearOverlay();
    setStatusType("neutral");
    setStatusMessage("Extracting facial embedding & saving to MongoDB...");

    try {
      const { blob, filename } = await getImagePayload();
      const formData = new FormData();
      formData.append("name", name.trim());
      formData.append("image", blob, filename);

      const res = await fetch(`${API_BASE}/faces/`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        setStatusType("error");
        setStatusMessage(data.error || "Enrollment failed.");
      } else {
        setStatusType("success");
        setStatusMessage(`Enrolled '${data.name}' into MongoDB.`);
        setName("");
      }
    } catch (err) {
      setStatusType("error");
      setStatusMessage(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRecognize = async () => {
    setLoading(true);
    setDetectedFaces([]);
    clearOverlay();
    setStatusType("neutral");
    setStatusMessage("Scanning face against MongoDB...");

    try {
      const { blob, filename, width, height, mirrored } = await getImagePayload();
      const formData = new FormData();
      formData.append("image", blob, filename);

      const res = await fetch(`${API_BASE}/recognize/`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        setStatusType("error");
        setStatusMessage(data.error || "Recognition request failed.");
        return;
      }

      const faces = data.faces || [];
      setDetectedFaces(faces);
      drawBoundingBoxes(faces, width, height, mirrored);

      if (data.status === "ok") {
        setStatusType("success");
        setStatusMessage(data.message);
      } else {
        setStatusType("error");
        setStatusMessage(data.message || "Cannot recognize");
      }
    } catch (err) {
      setStatusType("error");
      setStatusMessage(err.message);
    } finally {
      setLoading(false);
    }
  };

  const bannerTheme = {
    neutral: {
      bg: "rgba(30, 41, 59, 0.8)",
      border: "#334155",
      text: "#cbd5e1",
      dot: "#64748b",
    },
    success: {
      bg: "rgba(6, 78, 59, 0.55)",
      border: "#10b981",
      text: "#a7f3d0",
      dot: "#10b981",
    },
    error: {
      bg: "rgba(127, 29, 29, 0.55)",
      border: "#ef4444",
      text: "#fecaca",
      dot: "#ef4444",
    },
  }[statusType];

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "2.5rem 1rem",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 780,
          background: "rgba(15, 23, 42, 0.85)",
          backdropFilter: "blur(14px)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 20,
          padding: "2rem",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.6)",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
        }}
      >
        {/* Header Section — Clear vertical separation from controls */}
        <header style={{ textAlign: "center", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <span
            style={{
              alignSelf: "center",
              fontSize: "0.75rem",
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "#60a5fa",
              background: "rgba(37, 99, 235, 0.15)",
              border: "1px solid rgba(96, 165, 250, 0.3)",
              padding: "0.25rem 0.75rem",
              borderRadius: 999,
            }}
          >
            AI Identity Verification
          </span>
          <h1
            style={{
              fontSize: "1.9rem",
              fontWeight: 800,
              letterSpacing: "-0.02em",
              color: "#f8fafc",
              lineHeight: 1.2,
              marginTop: "0.25rem",
            }}
          >
            Face Recognition System
          </h1>
          <p style={{ fontSize: "0.95rem", color: "#94a3b8" }}>
            Enroll faces into MongoDB or scan live to verify identity
          </p>
        </header>

        {/* Segmented Control Switcher */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.5rem",
            background: "#0b1120",
            padding: "0.35rem",
            borderRadius: 12,
            border: "1px solid #1e293b",
          }}
        >
          <button
            type="button"
            onClick={() => handleModeSwitch("webcam")}
            style={{
              padding: "0.65rem 1rem",
              borderRadius: 9,
              border: "none",
              background: mode === "webcam" ? "#2563eb" : "transparent",
              color: mode === "webcam" ? "#ffffff" : "#94a3b8",
              fontWeight: 600,
              fontSize: "0.95rem",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            Live Webcam
          </button>
          <button
            type="button"
            onClick={() => handleModeSwitch("upload")}
            style={{
              padding: "0.65rem 1rem",
              borderRadius: 9,
              border: "none",
              background: mode === "upload" ? "#2563eb" : "transparent",
              color: mode === "upload" ? "#ffffff" : "#94a3b8",
              fontWeight: 600,
              fontSize: "0.95rem",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            Upload Photo
          </button>
        </div>

        {/* Camera / Image Viewport */}
        <div
          style={{
            position: "relative",
            borderRadius: 14,
            overflow: "hidden",
            background: "#020617",
            border: "1px solid #1e293b",
            aspectRatio: "4 / 3",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {mode === "webcam" ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
            />
          ) : previewUrl ? (
            <img
              src={previewUrl}
              alt="Uploaded preview"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          ) : (
            <label
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.5rem",
                color: "#94a3b8",
                cursor: "pointer",
                padding: "2rem",
                textAlign: "center",
              }}
            >
              <span style={{ fontSize: "1.05rem", fontWeight: 600, color: "#e2e8f0" }}>
                Click to select a photo
              </span>
              <span style={{ fontSize: "0.85rem", color: "#64748b" }}>
                Supports JPG, PNG, or WEBP
              </span>
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
            </label>
          )}

          <canvas
            ref={overlayCanvasRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
            }}
          />
          <canvas ref={captureCanvasRef} style={{ display: "none" }} />
        </div>

        {mode === "upload" && previewUrl && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "-0.75rem" }}>
            <label
              style={{
                fontSize: "0.85rem",
                color: "#60a5fa",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Choose a different photo
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
            </label>
          </div>
        )}

        {/* Status Indicator Banner */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.6rem",
            padding: "0.9rem 1.25rem",
            borderRadius: 12,
            border: `1px solid ${bannerTheme.border}`,
            background: bannerTheme.bg,
            color: bannerTheme.text,
            fontWeight: 600,
            fontSize: "0.95rem",
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: bannerTheme.dot,
              flexShrink: 0,
            }}
          />
          <span>{statusMessage}</span>
        </div>

        {/* Detected Faces List */}
        {detectedFaces.length > 0 && (
          <div style={{ display: "grid", gap: "0.6rem" }}>
            {detectedFaces.map((face, idx) => (
              <div
                key={idx}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "0.75rem 1rem",
                  borderRadius: 10,
                  background: "#0b1120",
                  border: "1px solid #1e293b",
                  borderLeft: `4px solid ${face.recognized ? "#10b981" : "#ef4444"}`,
                }}
              >
                <span style={{ fontWeight: 600, color: face.recognized ? "#34d399" : "#f87171" }}>
                  {face.recognized ? face.name : "Cannot recognize"}
                </span>
                {typeof face.score === "number" && (
                  <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                    Score: {face.score.toFixed(3)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Controls Section: Primary Recognize Action + Enrollment Card */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
            paddingTop: "0.5rem",
            borderTop: "1px solid #1e293b",
          }}
        >
          <button
            type="button"
            onClick={handleRecognize}
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.9rem 1.5rem",
              fontSize: "1rem",
              fontWeight: 700,
              borderRadius: 12,
              border: "none",
              background: loading ? "#1d4ed8" : "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
              color: "#ffffff",
              cursor: loading ? "not-allowed" : "pointer",
              boxShadow: "0 10px 20px -6px rgba(37, 99, 235, 0.5)",
              opacity: loading ? 0.75 : 1,
            }}
          >
            {loading ? "Processing..." : "Recognize Face"}
          </button>

          <form
            onSubmit={handleEnroll}
            style={{
              display: "flex",
              gap: "0.65rem",
              flexWrap: "wrap",
            }}
          >
            <input
              type="text"
              placeholder="Enter person's full name to enroll..."
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
              style={{
                flex: "1 1 260px",
                padding: "0.85rem 1rem",
                fontSize: "0.95rem",
                borderRadius: 12,
                border: "1px solid #334155",
                background: "#0b1120",
                color: "#f8fafc",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                flex: "0 0 auto",
                padding: "0.85rem 1.5rem",
                fontSize: "0.95rem",
                fontWeight: 700,
                borderRadius: 12,
                border: "none",
                background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                color: "#ffffff",
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: "0 10px 20px -6px rgba(16, 185, 129, 0.4)",
                opacity: loading ? 0.75 : 1,
              }}
            >
              Enroll to Database
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}