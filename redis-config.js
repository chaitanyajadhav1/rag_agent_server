import IORedis from 'ioredis';

/**
 * Create Redis connection for Upstash (free tier)
 * Supports both REST URL and standard Redis URL formats
 */
export function createRedisConnection() {
  try {
    // Option 1: Upstash REST URL (primary method)
    if (process.env.UPSTASH_REDIS_REST_URL) {
      console.log('[Redis] Connecting to Upstash Redis...');
      
      // Clean the URL (remove any quotes)
      let restUrl = process.env.UPSTASH_REDIS_REST_URL.trim();
      restUrl = restUrl.replace(/^["']|["']$/g, ''); // Remove surrounding quotes
      
      // Parse URL - Upstash gives redis:// format, convert to use with ioredis
      const urlObj = new URL(restUrl);
      
      const connection = new IORedis({
        host: urlObj.hostname,
        port: parseInt(urlObj.port) || 6379,
        password: process.env.UPSTASH_REDIS_REST_TOKEN?.replace(/^["']|["']$/g, ''),
        tls: {
          rejectUnauthorized: false
        },
        maxRetriesPerRequest: null,
        enableReadyCheck: false,
        // CRITICAL: Disable CLIENT SETINFO to prevent errors with Upstash
        showFriendlyErrorStack: false,
        enableOfflineQueue: true,
        // Don't send CLIENT SETNAME/SETINFO commands
        connectionName: null,
        retryStrategy: (times) => {
          const delay = Math.min(times * 50, 2000);
          return delay;
        },
        reconnectOnError: (err) => {
          // Only reconnect on network errors, not command errors
          const targetError = /READONLY|ETIMEDOUT|ECONNRESET|ENOTFOUND/;
          if (targetError.test(err.message)) {
            console.log('[Redis] Reconnecting due to:', err.message);
            return true;
          }
          // Ignore CLIENT SETINFO errors - don't reconnect
          if (err.message.includes('CLIENT SETINFO')) {
            return false;
          }
          return false;
        }
      });

      connection.on('connect', () => {
        console.log('[Redis] ✓ Connected to Upstash Redis');
      });

      connection.on('error', (err) => {
        // Suppress CLIENT SETINFO error logs
        if (!err.message.includes('CLIENT SETINFO')) {
          console.error('[Redis] Connection error:', err.message);
        }
      });

      return connection;
    }

    // Option 2: Standard Redis URL
    if (process.env.REDIS_URL) {
      console.log('[Redis] Connecting using REDIS_URL...');
      return new IORedis(process.env.REDIS_URL, {
        maxRetriesPerRequest: null,
        enableReadyCheck: false,
        connectionName: null, // Disable CLIENT SETINFO
        tls: process.env.REDIS_TLS === 'true' ? { rejectUnauthorized: false } : undefined
      });
    }

    // Option 3: Manual configuration (local development)
    console.log('[Redis] Using manual configuration...');
    return new IORedis({
      host: process.env.REDIS_HOST || '127.0.0.1',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      password: process.env.REDIS_PASSWORD || undefined,
      maxRetriesPerRequest: null,
      enableReadyCheck: false,
      connectionName: null // Disable CLIENT SETINFO
    });

  } catch (error) {
    console.error('[Redis] Failed to create connection:', error);
    throw error;
  }
}

export const redisConnection = createRedisConnection();