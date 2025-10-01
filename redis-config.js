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
      
      // Parse Upstash REST URL
      const restUrl = process.env.UPSTASH_REDIS_REST_URL;
      const url = new URL(restUrl.replace('redis://', 'http://'));
      
      const connection = new IORedis({
        host: url.hostname,
        port: parseInt(url.port) || 6379,
        password: process.env.UPSTASH_REDIS_REST_TOKEN,
        tls: {
          rejectUnauthorized: false
        },
        maxRetriesPerRequest: null,
        enableReadyCheck: false,
        retryStrategy: (times) => {
          const delay = Math.min(times * 50, 2000);
          return delay;
        },
        reconnectOnError: (err) => {
          console.log('[Redis] Reconnect on error:', err.message);
          return true;
        }
      });

      connection.on('connect', () => {
        console.log('[Redis] ✓ Connected to Upstash Redis');
      });

      connection.on('error', (err) => {
        console.error('[Redis] Connection error:', err.message);
      });

      return connection;
    }

    // Option 2: Standard Redis URL
    if (process.env.REDIS_URL) {
      console.log('[Redis] Connecting using REDIS_URL...');
      return new IORedis(process.env.REDIS_URL, {
        maxRetriesPerRequest: null,
        enableReadyCheck: false,
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
      enableReadyCheck: false
    });

  } catch (error) {
    console.error('[Redis] Failed to create connection:', error);
    throw error;
  }
}

// Export singleton connection
export const redisConnection = createRedisConnection();