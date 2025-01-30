import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null); // To store the uploaded file
  const [downloadUrl, setDownloadUrl] = useState(""); // To store the mask download URL
  const [loading, setLoading] = useState(false); // To show a loading indicator

  // Handle file input change
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setDownloadUrl(""); // Reset the download URL if a new file is uploaded
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      alert("Please upload an image before submitting.");
      return;
    }

    setLoading(true); // Show loading indicator

    const formData = new FormData();
    formData.append("image", file);

    try {
      // Send the image to the backend
      const response = await axios.post("https://segmentationbackend.onrender.com/api/upload/", formData, {
        responseType: "blob", // Important for receiving the mask as a file
      });

      // Create a URL for the returned blob (mask image)
      const blob = new Blob([response.data], { type: "image/png" });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("An error occurred while processing the image.");
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <h1 className="text-2xl font-bold text-gray-700 mb-6">Image Segmentation App</h1>
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md bg-white shadow-md rounded-lg p-6 space-y-4"
      >
        {/* File Input */}
        <div>
          <label className="block text-gray-700 font-medium mb-2">
            Upload an Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border file:border-gray-300 file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? "Processing..." : "Submit"}
        </button>
      </form>

      {/* Download Button */}
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
