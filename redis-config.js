import { Redis } from '@upstash/redis';

/**
 * Create Redis connection for Upstash using REST API
 * Works with both serverless and traditional environments
 */
export function createRedisConnection() {
  try {
    // Check if Upstash credentials are available
    const url = process.env.UPSTASH_REDIS_REST_URL?.replace(/^["']|["']$/g, '');
    const token = process.env.UPSTASH_REDIS_REST_TOKEN?.replace(/^["']|["']$/g, '');

    if (!url || !token) {
      console.error('[Redis] Missing Upstash credentials!');
      console.error('[Redis] UPSTASH_REDIS_REST_URL:', url ? 'SET' : 'MISSING');
      console.error('[Redis] UPSTASH_REDIS_REST_TOKEN:', token ? 'SET' : 'MISSING');
      throw new Error('UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required');
    }

    console.log('[Redis] Connecting to Upstash Redis via REST...');
    console.log('[Redis] URL:', url);

    const redis = new Redis({
      url: url,
      token: token,
    });

    console.log('[Redis] ✓ Connected to Upstash Redis');
    return redis;

  } catch (error) {
    console.error('[Redis] Failed to create connection:', error.message);
    throw error;
  }
}

export const redisConnection = createRedisConnection();