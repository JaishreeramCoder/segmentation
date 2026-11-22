import React, { useEffect, useState } from "react";
import axios from "axios";

const BACKEND_URL = "https://segmentationbackend.onrender.com";

function App() {
  const [file, setFile] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    axios
      .get(`${BACKEND_URL}/api/health/`, { timeout: 120000 })
      .then(() => setBackendReady(true))
      .catch((error) => {
        console.error("Backend health check failed:", error);
      });
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setDownloadUrl("");
  };

  const handleSubmit = async (event) => {
    event.preventEvent();
    if (!file) {
      alert("Please upload an image before submitting.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/upload/`,
        formData,
        { responseType: "blob" }
      );

      const blob = new Blob([response.data], {
        type: "image/png",
      });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("An error occurred while processing the image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <h1 className="text-2xl font-bold text-gray-700 mb-6">
        Image Segmentation App
      </h1>

      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md bg-white shadow-md rounded-lg p-6 space-y-4"
      >
        <div>
          <label className="block text-gray-700 font-medium mb-2">
            Upload an Image
          </label>

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-4 file:px-4 file:rounded-lg file:border file:border-gray-300 file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200"
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 disabled:opacity-50"
          disabled={loading || !backendReady}
        >
          {loading
            ? "Processing..."
            : backendReady
              ? "Submit"
              : "Waking up backend..."}
        </button>
      </form>

      {downloadUrl && (
        <div className="mt-6">
          <a
            href={downloadUrl}
            download="segmentation_mask.png"
            className="inline-block bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600"
          >
            Download Mask
          </a>
        </div>
      )}
    </div>
  );
}

export default App;
