const { v2: cloudinary } = require('cloudinary');

function ensureCloudinaryConfigured() {
  const { CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET } = process.env;
  if (!CLOUDINARY_CLOUD_NAME || !CLOUDINARY_API_KEY || !CLOUDINARY_API_SECRET) {
    throw new Error('Cloudinary credentials are missing. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET.');
  }

  cloudinary.config({
    cloud_name: CLOUDINARY_CLOUD_NAME,
    api_key: CLOUDINARY_API_KEY,
    api_secret: CLOUDINARY_API_SECRET,
  });
}

/**
 * Upload a local audio file to Cloudinary and return the secure URL.
 * @param {string} filePath
 * @returns {Promise<string>}
 */
async function uploadAudioToCloudinary(filePath) {
  if (!filePath) {
    throw new Error('uploadAudioToCloudinary requires a file path.');
  }

  ensureCloudinaryConfigured();
  console.log('☁️  Uploading audio to Cloudinary:', filePath);

  const result = await cloudinary.uploader.upload(filePath, {
    resource_type: 'video', // Cloudinary treats audio uploads as video resources
    folder: 'audioshake-uploads',
  });

  if (!result?.secure_url) {
    throw new Error('Cloudinary upload did not return a secure URL.');
  }

  console.log('☁️  Cloudinary upload complete:', result.secure_url);
  return result.secure_url;
}

module.exports = {
  uploadAudioToCloudinary,
};

