import { useEffect, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000/api";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [name, setName] = useState("");
  const [statusMessage, setStatusMessage] = useState("Camera ready.");
  const [statusType, setStatusType] = useState("neutral"); // "neutral" | "success" | "error"
  const [loading, setLoading] = useState(false);

  // Start webcam on mount
  useEffect(() => {
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
  }, []);

  // Capture current video frame as a JPEG Blob
  const captureFrameBlob = () =>
    new Promise((resolve, reject) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
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
          if (blob) resolve(blob);
          else reject(new Error("Failed to capture image frame."));
        },
        "image/jpeg",
        0.92
      );
    });

  // POST /api/faces/ -> Enroll person into MongoDB
  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setStatusType("error");
      setStatusMessage("Please enter a name before enrolling.");
      return;
    }

    setLoading(true);
    setStatusType("neutral");
    setStatusMessage("Enrolling face...");

    try {
      const imageBlob = await captureFrameBlob();
      const formData = new FormData();
      formData.append("name", name.trim());
      formData.append("image", imageBlob, "enroll.jpg");

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
        setStatusMessage(`Enrolled ${data.name} into database.`);
        setName("");
      }
    } catch (err) {
      setStatusType("error");
      setStatusMessage("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // POST /api/recognize/ -> Identify face or show "Cannot recognize"
  const handleRecognize = async () => {
    setLoading(true);
    setStatusType("neutral");
    setStatusMessage("Scanning face...");

    try {
      const imageBlob = await captureFrameBlob();
      const formData = new FormData();
      formData.append("image", imageBlob, "scan.jpg");

      const res = await fetch(`${API_BASE}/recognize/`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        setStatusType("error");
        setStatusMessage(data.error || "Recognition failed.");
      } else if (data.status === "ok") {
        setStatusType("success");
        setStatusMessage(data.message);
      } else {
        // Handles data.status === "unknown" -> "Cannot recognize"
        setStatusType("error");
        setStatusMessage(data.message || "Cannot recognize");
      }
    } catch (err) {
      setStatusType("error");
      setStatusMessage("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const bannerColors = {
    neutral: { background: "#1e293b", color: "#e2e8f0", border: "#334155" },
    success: { background: "#064e3b", color: "#a7f3d0", border: "#059669" },
    error: { background: "#7f1d1d", color: "#fecaca", border: "#dc2626" },
  };

  return (
    <div style={{ maxWidth: 680, margin: "2rem auto", fontFamily: "system-ui, sans-serif", padding: "0 1rem" }}>
      <h1 style={{ textAlign: "center", marginBottom: "1rem" }}>Face Recognition System</h1>

      <div style={{ position: "relative", borderRadius: 12, overflow: "hidden", background: "#000" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: "100%", display: "block", transform: "scaleX(-1)" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div
        style={{
          marginTop: "1rem",
          padding: "0.9rem 1rem",
          borderRadius: 8,
          border: `1px solid ${bannerColors[statusType].border}`,
          background: bannerColors[statusType].background,
          color: bannerColors[statusType].color,
          fontWeight: 600,
          textAlign: "center",
        }}
      >
        {statusMessage}
      </div>

      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={handleRecognize}
          disabled={loading}
          style={{
            flex: "1 1 180px",
            padding: "0.75rem 1.25rem",
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

        <form onSubmit={handleEnroll} style={{ display: "flex", flex: "2 1 300px", gap: "0.5rem" }}>
          <input
            type="text"
            placeholder="Enter name to register..."
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={loading}
            style={{
              flex: 1,
              padding: "0.75rem",
              fontSize: "1rem",
              borderRadius: 8,
              border: "1px solid #cbd5e1",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "0.75rem 1.25rem",
              fontSize: "1rem",
              fontWeight: 600,
              borderRadius: 8,
              border: "none",
              background: "#10b981",
              color: "#fff",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            Enroll
          </button>
        </form>
      </div>
    </div>
  );
}