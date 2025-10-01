import IORedis from 'ioredis';

/**
 * Create Redis connection for Upstash with BullMQ compatibility
 * This configuration prevents CLIENT SETINFO errors
 */
export function createRedisConnection() {
  try {
    const restUrl = process.env.UPSTASH_REDIS_REST_URL?.replace(/^["']|["']$/g, '');
    const token = process.env.UPSTASH_REDIS_REST_TOKEN?.replace(/^["']|["']$/g, '');

    if (!restUrl || !token) {
      console.error('[Redis] Missing Upstash credentials');
      console.error('[Redis] URL:', restUrl ? '✓' : '✗ MISSING');
      console.error('[Redis] Token:', token ? '✓' : '✗ MISSING');
      throw new Error('UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN required');
    }

    // Convert HTTPS URL to Redis URL format
    // Upstash gives: https://above-jackass-16825.upstash.io
    // We need: rediss://default:TOKEN@above-jackass-16825.upstash.io:6379
    const hostname = restUrl.replace(/^https?:\/\//, '');
    
    console.log('[Redis] Connecting to Upstash...');
    console.log('[Redis] Host:', hostname);

    const connection = new IORedis({
      host: hostname,
      port: 6379,
      password: token,
      username: 'default',
      family: 4,
      tls: {
        rejectUnauthorized: false
      },
      // CRITICAL: These settings prevent CLIENT SETINFO errors
      enableReadyCheck: false,
      maxRetriesPerRequest: null,
      enableOfflineQueue: true,
      showFriendlyErrorStack: false,
      connectionName: null, // Prevents CLIENT SETINFO/SETNAME
      
      retryStrategy: (times) => {
        if (times > 10) {
          console.error('[Redis] Max retries reached');
          return null;
        }
        const delay = Math.min(times * 200, 3000);
        return delay;
      },
      
      reconnectOnError: (err) => {
        // Only reconnect on real network errors, not command errors
        const networkErrors = /READONLY|ETIMEDOUT|ECONNRESET|ENOTFOUND|ECONNREFUSED/;
        if (networkErrors.test(err.message)) {
          return 1; // reconnect
        }
        // Ignore CLIENT SETINFO errors completely
        if (err.message.includes('CLIENT SETINFO') || 
            err.message.includes('CLIENT SETNAME')) {
          return false; // don't reconnect
        }
        return false;
      }
    });

    // Suppress CLIENT SETINFO error logs
    connection.on('error', (err) => {
      if (!err.message.includes('CLIENT SETINFO') && 
          !err.message.includes('CLIENT SETNAME')) {
        console.error('[Redis] Error:', err.message);
      }
    });

    connection.on('connect', () => {
      console.log('[Redis] ✓ Connected to Upstash Redis');
    });

    connection.on('ready', () => {
      console.log('[Redis] ✓ Ready for commands');
    });

    return connection;

  } catch (error) {
    console.error('[Redis] Failed to create connection:', error.message);
    throw error;
  }
}

export const redisConnection = createRedisConnection();