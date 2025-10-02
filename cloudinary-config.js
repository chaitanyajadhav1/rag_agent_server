import cloudinaryLib from 'cloudinary';
import dotenv from 'dotenv';

dotenv.config();

const { v2: cloudinary } = cloudinaryLib;

// Configure Cloudinary with your credentials
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true // Use HTTPS URLs
});

// Verify configuration on startup
const verifyCloudinaryConfig = () => {
  const requiredVars = [
    'CLOUDINARY_CLOUD_NAME',
    'CLOUDINARY_API_KEY', 
    'CLOUDINARY_API_SECRET'
  ];

  const missing = requiredVars.filter(varName => !process.env[varName]);

  if (missing.length > 0) {
    console.error('❌ Missing Cloudinary environment variables:', missing.join(', '));
    console.error('Please add them to your .env file');
    return false;
  }

  console.log('✓ Cloudinary configured successfully');
  console.log(`  Cloud Name: ${process.env.CLOUDINARY_CLOUD_NAME}`);
  console.log(`  API Key: ${process.env.CLOUDINARY_API_KEY.substring(0, 8)}...`);
  return true;
};

// Run verification
verifyCloudinaryConfig();

export default cloudinary;