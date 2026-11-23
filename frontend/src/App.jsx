import { useEffect, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000/api";

export default function App() {
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);

  const [mode, setMode] = useState("webcam"); // "webcam" | "upload"
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [name, setName] = useState("");
  const [personId, setPersonId] = useState("");

  const [statusMessage, setStatusMessage] = useState("Camera ready.");
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
        setStatusMessage("Camera ready.");
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

  // Clear bounding box overlay when switching modes or images
  const clearOverlay = () => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  // Draw bounding boxes from backend `faces` array onto the overlay canvas
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
      // Flip X coordinate when rendering over mirrored webcam preview
      const x = mirror ? sourceWidth - rawX - w : rawX;

      const color = face.recognized ? "#10b981" : "#ef4444";
      const label = face.recognized
        ? `${face.name}${typeof face.score === "number" ? ` (${face.score.toFixed(2)})` : ""}`
        : "Cannot recognize";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.font = "bold 15px system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelY = y > 26 ? y - 26 : y + h;

      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, textWidth + 14, 26);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + 7, labelY + 18);
    });
  };

  // Handle switching between Webcam and Image Upload
  const handleModeSwitch = (nextMode) => {
    setMode(nextMode);
    setDetectedFaces([]);
    clearOverlay();
    if (nextMode === "webcam") {
      setUploadedFile(null);
      setPreviewUrl(null);
    } else {
      setStatusType("neutral");
      setStatusMessage("Select an image file to enroll or recognize.");
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
    setStatusMessage(`Loaded ${file.name}`);
  };

  // Obtain image Blob + natural dimensions from either webcam or uploaded file
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

  // POST /api/faces/ -> Enroll face in MongoDB
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
    setStatusMessage("Extracting embedding & saving to MongoDB...");

    try {
      const { blob, filename } = await getImagePayload();
      const formData = new FormData();
      formData.append("name", name.trim());
      if (personId.trim()) {
        formData.append("person_id", personId.trim());
      }
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
        setPersonId("");
      }
    } catch (err) {
      setStatusType("error");
      setStatusMessage(err.message);
    } finally {
      setLoading(false);
    }
  };

  // POST /api/recognize/ -> Match against MongoDB or show "Cannot recognize"
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

  const bannerStyles = {
    neutral: { background: "#1e293b", color: "#e2e8f0", border: "#334155" },
    success: { background: "#064e3b", color: "#a7f3d0", border: "#059669" },
    error: { background: "#7f1d1d", color: "#fecaca", border: "#dc2626" },
  };

  return (
    <div style={{ maxWidth: 740, margin: "1.5rem auto", fontFamily: "system-ui, sans-serif", padding: "0 1rem" }}>
      <h1 style={{ textAlign: "center", marginBottom: "0.75rem" }}>Face Recognition System</h1>

      {/* Mode Switcher: Live Webcam vs Image File */}
      <div style={{ display: "flex", justifyContent: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <button
          type="button"
          onClick={() => handleModeSwitch("webcam")}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: 8,
            border: "1px solid #334155",
            background: mode === "webcam" ? "#2563eb" : "#1e293b",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Live Webcam
        </button>
        <button
          type="button"
          onClick={() => handleModeSwitch("upload")}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: 8,
            border: "1px solid #334155",
            background: mode === "upload" ? "#2563eb" : "#1e293b",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Upload Photo
        </button>
      </div>

      {/* Viewport with Bounding Box Canvas Overlay */}
      <div
        style={{
          position: "relative",
          borderRadius: 12,
          overflow: "hidden",
          background: "#0f172a",
          minHeight: 320,
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
            style={{ width: "100%", display: "block", transform: "scaleX(-1)" }}
          />
        ) : previewUrl ? (
          <img
            src={previewUrl}
            alt="Uploaded preview"
            style={{ width: "100%", display: "block", objectFit: "contain" }}
          />
        ) : (
          <label
            style={{
              padding: "2rem",
              color: "#94a3b8",
              cursor: "pointer",
              textAlign: "center",
              width: "100%",
            }}
          >
            Click to select a JPG/PNG photo
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
        <div style={{ marginTop: "0.5rem", textAlign: "right" }}>
          <label style={{ fontSize: "0.9rem", color: "#2563eb", cursor: "pointer", fontWeight: 600 }}>
            Change Photo
            <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
          </label>
        </div>
      )}

      {/* Status Banner */}
      <div
        style={{
          marginTop: "1rem",
          padding: "0.9rem 1rem",
          borderRadius: 8,
          border: `1px solid ${bannerStyles[statusType].border}`,
          background: bannerStyles[statusType].background,
          color: bannerStyles[statusType].color,
          fontWeight: 600,
          textAlign: "center",
        }}
      >
        {statusMessage}
      </div>

      {/* Per-Face Breakdown */}
      {detectedFaces.length > 0 && (
        <div style={{ display: "grid", gap: "0.5rem", marginTop: "0.75rem" }}>
          {detectedFaces.map((face, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.65rem 1rem",
                borderRadius: 8,
                background: "#1e293b",
                color: "#f8fafc",
                borderLeft: `4px solid ${face.recognized ? "#10b981" : "#ef4444"}`,
              }}
            >
              <span style={{ fontWeight: 600 }}>
                {face.recognized ? face.name : "Cannot recognize"}
              </span>
              {typeof face.score === "number" && (
                <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                  Distance/Score: {face.score.toFixed(3)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Actions: Recognize Button & Enrollment Form */}
      <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
        <button
          type="button"
          onClick={handleRecognize}
          disabled={loading}
          style={{
            width: "100%",
            padding: "0.8rem 1.25rem",
            fontSize: "1rem",
            fontWeight: 600,
            borderRadius: 8,
            border: "none",
            background: "#2563eb",
            color: "#fff",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Processing..." : "Recognize Face"}
        </button>

        <form
          onSubmit={handleEnroll}
          style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
        >
          <input
            type="text"
            placeholder="Full Name (required)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={loading}
            style={{
              flex: "2 1 200px",
              padding: "0.75rem",
              fontSize: "0.95rem",
              borderRadius: 8,
              border: "1px solid #cbd5e1",
            }}
          />
          <input
            type="text"
            placeholder="ID / USN (optional)"
            value={personId}
            onChange={(e) => setPersonId(e.target.value)}
            disabled={loading}
            style={{
              flex: "1 1 140px",
              padding: "0.75rem",
              fontSize: "0.95rem",
              borderRadius: 8,
              border: "1px solid #cbd5e1",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              flex: "1 1 120px",
              padding: "0.75rem 1.25rem",
              fontSize: "0.95rem",
              fontWeight: 600,
              borderRadius: 8,
              border: "none",
              background: "#10b981",
              color: "#fff",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            Enroll to DB
          </button>
        </form>
      </div>
    </div>
  );
}