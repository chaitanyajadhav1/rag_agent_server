import { Worker } from 'bullmq';
import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantVectorStore } from '@langchain/qdrant';
import { Document } from '@langchain/core/documents';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import OpenAI from 'openai';
import express from 'express';
import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ========== AI-POWERED DOCUMENT ANALYSIS ==========

async function analyzeDocumentWithAI(content, filename) {
  try {
    const prompt = `Analyze this document and extract key information.

Filename: ${filename}
Content: ${content.substring(0, 3000)}

Return JSON:
{
  "documentType": "invoice|bill_of_lading|packing_list|certificate|contract|manual|other",
  "confidence": 0.0-1.0,
  "language": "en|es|fr|de|zh|other",
  "summary": "brief summary",
  "keyEntities": {
    "companies": ["array"],
    "locations": ["array"],
    "dates": ["array"],
    "amounts": ["array"]
  },
  "topics": ["array of main topics"],
  "sentiment": "positive|neutral|negative|technical"
}`;

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { 
          role: 'system', 
          content: 'You are a document analysis expert. Return only valid JSON.' 
        },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 1000,
    });

    const analysis = JSON.parse(response.choices[0].message.content);
    return analysis;
  } catch (error) {
    console.warn('AI analysis failed, using fallback:', error.message);
    return {
      documentType: 'other',
      confidence: 0.3,
      language: 'en',
      summary: 'Analysis unavailable',
      keyEntities: { companies: [], locations: [], dates: [], amounts: [] },
      topics: [],
      sentiment: 'neutral'
    };
  }
}

// ========== PDF PROCESSING WORKER ==========

const pdfWorker = new Worker(
  'file-upload-queue',
  async (job) => {
    const startTime = Date.now();
    
    try {
      console.log(`[PDF Worker] Processing job ${job.id}`);
      const data = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;

      const { path: filePath, filename, collectionName, strategy, userId, documentId } = data;

      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      const fileSize = fs.statSync(filePath).size;
      console.log(`[PDF Worker] Loading ${filename} (${(fileSize / 1024).toFixed(2)} KB)`);

      // Load PDF
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      console.log(`[PDF Worker] Loaded ${docs.length} pages`);

      // AI-powered document analysis
      const fullText = docs.map(doc => doc.pageContent).join('\n');
      console.log(`[PDF Worker] Running AI analysis...`);
      const aiAnalysis = await analyzeDocumentWithAI(fullText, filename);
      console.log(`[PDF Worker] Detected: ${aiAnalysis.documentType} (confidence: ${aiAnalysis.confidence})`);

      // Text splitting with optimal parameters
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
        separators: ['\n\n', '\n', '. ', ' ', ''],
      });

      const splitDocs = await splitter.splitDocuments(docs);
      console.log(`[PDF Worker] Split into ${splitDocs.length} chunks`);

      // Enhance metadata with AI analysis
      const enrichedDocs = splitDocs.map((doc, index) => {
        return new Document({
          pageContent: doc.pageContent,
          metadata: {
            ...doc.metadata,
            source: filename,
            chunkIndex: index,
            totalChunks: splitDocs.length,
            documentId,
            userId,
            strategy,
            processedAt: new Date().toISOString(),
            
            // AI-enriched metadata
            documentType: aiAnalysis.documentType,
            language: aiAnalysis.language,
            confidence: aiAnalysis.confidence,
            summary: aiAnalysis.summary,
            topics: aiAnalysis.topics,
            sentiment: aiAnalysis.sentiment,
            entities: aiAnalysis.keyEntities,
            
            fileSize,
            processingVersion: '4.0.0-agent'
          },
        });
      });

      // Initialize embeddings
      const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-3-small',
        apiKey: process.env.OPENAI_API_KEY,
        batchSize: 100,
        stripNewLines: true,
      });

      console.log(`[PDF Worker] Creating vector store: ${collectionName}`);

      // Clear collection for 'document' strategy
      if (strategy === 'document') {
        try {
          const tmpStore = new QdrantVectorStore(embeddings, {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName,
            config: { checkCompatibility: false },
          });
          await tmpStore.client.deleteCollection(collectionName);
          console.log(`[PDF Worker] Cleared collection: ${collectionName}`);
        } catch {
          console.log(`[PDF Worker] Collection ${collectionName} did not exist`);
        }
      }

      // Create vector store
      const vectorStore = await QdrantVectorStore.fromDocuments(
        [],
        embeddings,
        {
          url: process.env.QDRANT_URL,
          apiKey: process.env.QDRANT_API_KEY,
          collectionName,
          config: { 
            checkCompatibility: false,
            vectors: {
              size: 1536,
              distance: 'Cosine'
            }
          },
        }
      );

      // Batch insert with progress tracking
      const batchSize = 25;
      let processedChunks = 0;
      
      for (let i = 0; i < enrichedDocs.length; i += batchSize) {
        const batch = enrichedDocs.slice(i, i + batchSize);
        try {
          await vectorStore.addDocuments(batch);
          processedChunks += batch.length;
          
          const progress = ((processedChunks / enrichedDocs.length) * 100).toFixed(1);
          console.log(`[PDF Worker] Progress: ${progress}% (${processedChunks}/${enrichedDocs.length})`);
        } catch (batchError) {
          console.error(`[PDF Worker] Batch ${Math.floor(i / batchSize) + 1} failed:`, batchError.message);
        }
      }

      // Cleanup
      try {
        fs.unlinkSync(filePath);
        console.log(`[PDF Worker] Cleaned up file: ${filePath}`);
      } catch (err) {
        console.warn(`[PDF Worker] Could not delete file: ${err.message}`);
      }

      const processingTime = Date.now() - startTime;
      console.log(`[PDF Worker] ✓ Completed in ${processingTime}ms`);

      return {
        success: true,
        filename,
        chunks: processedChunks,
        collectionName,
        strategy,
        aiAnalysis,
        processingTime,
        version: '4.0.0-agent'
      };

    } catch (err) {
      console.error(`[PDF Worker] Job ${job.id} failed:`, err);
      
      // Cleanup on failure
      try {
        const filePath = typeof job.data === 'string' ? JSON.parse(job.data).path : job.data.path;
        if (filePath && fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      } catch {}
      
      throw err;
    }
  },
  {
    concurrency: 3,
    connection: {
      host: process.env.REDIS_HOST || '127.0.0.1',
      port: parseInt(process.env.REDIS_PORT) || 6379,
    },
    limiter: {
      max: 10,
      duration: 60000,
    },
    defaultJobOptions: {
      removeOnComplete: 100,
      removeOnFail: 50,
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000,
      },
    }
  }
);

// ========== SHIPPING DOCUMENT WORKER ==========

async function analyzeShippingDocumentWithAI(content, filename) {
  try {
    const prompt = `Analyze this shipping/logistics document and extract structured information.

Filename: ${filename}
Content: ${content}

Return JSON:
{
  "documentType": "Commercial Invoice|Bill of Lading|Packing List|Certificate of Origin|Air Waybill|Customs Declaration|Insurance Certificate|Unknown",
  "confidence": 0.0-1.0,
  "extractedData": {
    "invoiceNumber": "string or null",
    "shipperName": "string or null",
    "shipperAddress": "string or null",
    "consigneeName": "string or null",
    "consigneeAddress": "string or null",
    "portOfLoading": "string or null",
    "portOfDischarge": "string or null",
    "totalValue": "string or null",
    "currency": "string or null",
    "incoterms": "string or null",
    "weight": "string or null",
    "vessel": "string or null"
  },
  "validation": {
    "isComplete": boolean,
    "missingFields": ["array"],
    "complianceScore": 0.0-1.0
  },
  "summary": "brief summary"
}`;

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { 
          role: 'system', 
          content: 'You are a shipping document analysis expert. Extract structured data accurately.' 
        },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 2000,
    });

    const analysis = JSON.parse(response.choices[0].message.content);
    return analysis;
  } catch (error) {
    console.warn('Shipping document AI analysis failed:', error.message);
    return {
      documentType: 'Unknown',
      confidence: 0.2,
      extractedData: {},
      validation: {
        isComplete: false,
        missingFields: ['Analysis failed'],
        complianceScore: 0.0
      },
      summary: 'Analysis unavailable'
    };
  }
}

const shippingWorker = new Worker(
  'shipping-document-queue',
  async (job) => {
    const startTime = Date.now();
    
    try {
      console.log(`[Shipping Worker] Processing job ${job.id}`);
      const data = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;

      const { path: filePath, filename, sessionId, documentId, userId } = data;

      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      console.log(`[Shipping Worker] Loading ${filename}`);
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();

      const fullContent = docs.map(doc => doc.pageContent).join('\n');
      console.log(`[Shipping Worker] Running AI analysis...`);
      
      const analysis = await analyzeShippingDocumentWithAI(fullContent, filename);
      console.log(`[Shipping Worker] Detected: ${analysis.documentType} (${analysis.confidence})`);

      // Store in vector database for retrieval
      const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-3-small',
        apiKey: process.env.OPENAI_API_KEY,
      });

      const collectionName = `shipping_docs_${userId}`;
      
      const analyzedDoc = new Document({
        pageContent: fullContent,
        metadata: {
          source: filename,
          sessionId,
          documentId,
          userId,
          type: 'shipping_document',
          analysis: analysis,
          processedAt: new Date().toISOString(),
          fileSize: fs.statSync(filePath).size,
          version: '4.0.0-agent'
        },
      });

      try {
        const vectorStore = await QdrantVectorStore.fromDocuments(
          [],
          embeddings,
          {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName,
            config: { checkCompatibility: false },
          }
        );

        await vectorStore.addDocuments([analyzedDoc]);
        console.log(`[Shipping Worker] Stored in collection: ${collectionName}`);
      } catch (error) {
        console.warn(`[Shipping Worker] Vector storage failed: ${error.message}`);
      }

      // Cleanup
      try {
        fs.unlinkSync(filePath);
        console.log(`[Shipping Worker] Cleaned up file`);
      } catch (err) {
        console.warn(`[Shipping Worker] Could not delete file: ${err.message}`);
      }

      const processingTime = Date.now() - startTime;
      console.log(`[Shipping Worker] ✓ Completed in ${processingTime}ms`);

      return {
        success: true,
        filename,
        sessionId,
        documentId,
        analysis,
        collectionName,
        processingTime,
        version: '4.0.0-agent'
      };

    } catch (err) {
      console.error(`[Shipping Worker] Job ${job.id} failed:`, err);
      
      try {
        const filePath = typeof job.data === 'string' ? JSON.parse(job.data).path : job.data.path;
        if (filePath && fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      } catch {}
      
      throw err;
    }
  },
  {
    concurrency: 2,
    connection: {
      host: process.env.REDIS_HOST || '127.0.0.1',
      port: parseInt(process.env.REDIS_PORT) || 6379,
    },
    limiter: {
      max: 5,
      duration: 60000,
    },
    defaultJobOptions: {
      removeOnComplete: 50,
      removeOnFail: 25,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 3000,
      },
    }
  }
);

// ========== EVENT HANDLERS ==========

pdfWorker.on('completed', (job, result) => {
  console.log(`[PDF Worker] ✓ Job ${job.id} completed: ${result.filename} (${result.chunks} chunks, ${result.processingTime}ms)`);
});

pdfWorker.on('failed', (job, err) => {
  console.error(`[PDF Worker] ✗ Job ${job?.id} failed: ${err.message}`);
});

pdfWorker.on('stalled', (job) => {
  console.warn(`[PDF Worker] ⚠ Job ${job.id} stalled`);
});

shippingWorker.on('completed', (job, result) => {
  console.log(`[Shipping Worker] ✓ Job ${job.id} completed: ${result.filename} (${result.analysis.documentType})`);
});

shippingWorker.on('failed', (job, err) => {
  console.error(`[Shipping Worker] ✗ Job ${job?.id} failed: ${err.message}`);
});

shippingWorker.on('stalled', (job) => {
  console.warn(`[Shipping Worker] ⚠ Job ${job.id} stalled`);
});

// ========== HEALTH MONITORING ==========

setInterval(() => {
  const pdfStatus = pdfWorker.isRunning() ? '✓ Running' : '✗ Stopped';
  const shippingStatus = shippingWorker.isRunning() ? '✓ Running' : '✗ Stopped';
  console.log(`[Workers] PDF: ${pdfStatus} | Shipping: ${shippingStatus}`);
}, 60000);

// ========== HEALTH CHECK API ==========

const workerApp = express();
const WORKER_PORT = process.env.WORKER_PORT || 8001;

workerApp.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    workers: [
      {
        name: 'PDF Processing',
        queue: 'file-upload-queue',
        status: pdfWorker.isRunning() ? 'running' : 'stopped',
        concurrency: 3,
        features: ['AI Document Analysis', 'Vector Embedding', 'Metadata Enrichment']
      },
      {
        name: 'Shipping Documents',
        queue: 'shipping-document-queue', 
        status: shippingWorker.isRunning() ? 'running' : 'stopped',
        concurrency: 2,
        features: ['AI Extraction', 'Compliance Validation', 'Data Structuring']
      }
    ],
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    version: '4.0.0-agent',
    architecture: 'Agent-Based with AI Enhancement'
  });
});

workerApp.get('/stats', async (req, res) => {
  try {
    const pdfQueue = pdfWorker.queue;
    const shippingQueue = shippingWorker.queue;
    
    const [pdfWaiting, pdfActive, pdfCompleted, pdfFailed] = await Promise.all([
      pdfQueue.getWaiting(),
      pdfQueue.getActive(),
      pdfQueue.getCompleted(),
      pdfQueue.getFailed()
    ]);
    
    const [shipWaiting, shipActive, shipCompleted, shipFailed] = await Promise.all([
      shippingQueue.getWaiting(),
      shippingQueue.getActive(),
      shippingQueue.getCompleted(),
      shippingQueue.getFailed()
    ]);

    res.json({
      pdfProcessing: {
        waiting: pdfWaiting.length,
        active: pdfActive.length,
        completed: pdfCompleted.length,
        failed: pdfFailed.length,
        total: pdfWaiting.length + pdfActive.length + pdfCompleted.length + pdfFailed.length
      },
      shippingDocuments: {
        waiting: shipWaiting.length,
        active: shipActive.length,
        completed: shipCompleted.length,
        failed: shipFailed.length,
        total: shipWaiting.length + shipActive.length + shipCompleted.length + shipFailed.length
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to fetch stats', 
      message: error.message 
    });
  }
});

workerApp.get('/jobs/pdf/recent', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const completed = await pdfWorker.queue.getCompleted(0, limit - 1);
    const failed = await pdfWorker.queue.getFailed(0, limit - 1);
    
    res.json({
      completed: completed.map(job => ({
        id: job.id,
        name: job.name,
        data: job.data,
        returnvalue: job.returnvalue,
        finishedOn: job.finishedOn
      })),
      failed: failed.map(job => ({
        id: job.id,
        name: job.name,
        data: job.data,
        failedReason: job.failedReason,
        failedOn: job.finishedOn
      }))
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch jobs' });
  }
});

workerApp.get('/jobs/shipping/recent', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const completed = await shippingWorker.queue.getCompleted(0, limit - 1);
    const failed = await shippingWorker.queue.getFailed(0, limit - 1);
    
    res.json({
      completed: completed.map(job => ({
        id: job.id,
        name: job.name,
        data: job.data,
        returnvalue: job.returnvalue,
        finishedOn: job.finishedOn
      })),
      failed: failed.map(job => ({
        id: job.id,
        name: job.name,
        data: job.data,
        failedReason: job.failedReason,
        failedOn: job.finishedOn
      }))
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch jobs' });
  }
});

workerApp.listen(WORKER_PORT, () => {
  console.log('========================================');
  console.log('FreightChat Pro Workers Started');
  console.log('========================================');
  console.log(`Version: 4.0.0-agent`);
  console.log(`Architecture: Agent-Based AI Processing`);
  console.log(`Health Check: http://localhost:${WORKER_PORT}/health`);
  console.log(`Statistics: http://localhost:${WORKER_PORT}/stats`);
  console.log(`Recent PDF Jobs: http://localhost:${WORKER_PORT}/jobs/pdf/recent`);
  console.log(`Recent Shipping Jobs: http://localhost:${WORKER_PORT}/jobs/shipping/recent`);
  console.log('========================================');
  console.log('Active Workers:');
  console.log('  ✓ PDF Processing (3 concurrent)');
  console.log('    - AI document analysis');
  console.log('    - Vector embedding generation');
  console.log('    - Intelligent metadata extraction');
  console.log('  ✓ Shipping Documents (2 concurrent)');
  console.log('    - AI data extraction');
  console.log('    - Compliance validation');
  console.log('    - Structured data output');
  console.log('========================================');
  console.log('Ready to process jobs...');
});

// ========== GRACEFUL SHUTDOWN ==========

async function gracefulShutdown() {
  console.log('\n[Shutdown] Gracefully shutting down workers...');
  
  try {
    await Promise.all([
      pdfWorker.close(),
      shippingWorker.close()
    ]);
    console.log('[Shutdown] ✓ Workers closed successfully');
    process.exit(0);
  } catch (error) {
    console.error('[Shutdown] ✗ Error during shutdown:', error);
    process.exit(1);
  }
}

process.on('SIGINT', gracefulShutdown);
process.on('SIGTERM', gracefulShutdown);

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('[Fatal] Uncaught exception:', error);
  gracefulShutdown();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[Fatal] Unhandled rejection at:', promise, 'reason:', reason);
  gracefulShutdown();
});

console.log('Workers initialized and ready for agent-based processing!');