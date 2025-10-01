import { redisConnection } from './redis-config.js';

async function testRedis() {
  try {
    await redisConnection.set('test', 'Hello Upstash!');
    const value = await redisConnection.get('test');
    console.log('Redis test successful:', value);
    await redisConnection.del('test');
    process.exit(0);
  } catch (error) {
    console.error('Redis test failed:', error);
    process.exit(1);
  }
}

testRedis();