import { Worker } from 'bullmq';
import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantVectorStore } from '@langchain/qdrant';
import { Document } from '@langchain/core/documents';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import OpenAI from 'openai';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import { redisConnection } from './redis-config.js';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

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

// ========== INVOICE ANALYSIS WITH AI ==========

async function analyzeInvoiceWithAI(content, filename) {
  try {
    if (!content || content.trim().length < 50) {
      console.error('[Invoice AI] ERROR: Content too short or empty');
      throw new Error('PDF content is empty or too short to analyze');
    }

    console.log(`[Invoice AI] Starting analysis for: ${filename}`);
    console.log(`[Invoice AI] Content length: ${content.length} characters`);
    
    const contentToAnalyze = content.substring(0, 8000);
    
    const prompt = `Analyze this invoice/shipping document and extract ALL relevant information.

Filename: ${filename}

Document Content:
${contentToAnalyze}

Extract and return JSON with ALL available fields:
{
  "documentType": "Commercial Invoice|Proforma Invoice|Bill of Lading|Packing List|Certificate of Origin|Air Waybill|Sea Waybill|Customs Declaration|Insurance Certificate|Delivery Order|Other",
  "confidence": 0.0-1.0,
  "invoiceDetails": {
    "invoiceNumber": "string or null",
    "invoiceDate": "YYYY-MM-DD or null",
    "dueDate": "YYYY-MM-DD or null",
    "poNumber": "string or null",
    "currency": "USD|EUR|INR|etc or null"
  },
  "seller": {
    "name": "string or null",
    "address": "string or null",
    "city": "string or null",
    "country": "string or null",
    "taxId": "string or null",
    "contact": "string or null",
    "email": "string or null",
    "phone": "string or null"
  },
  "buyer": {
    "name": "string or null",
    "address": "string or null",
    "city": "string or null",
    "country": "string or null",
    "taxId": "string or null",
    "contact": "string or null",
    "email": "string or null",
    "phone": "string or null"
  },
  "shipmentDetails": {
    "portOfLoading": "string or null",
    "portOfDischarge": "string or null",
    "placeOfDelivery": "string or null",
    "countryOfOrigin": "string or null",
    "countryOfDestination": "string or null",
    "vessel": "string or null",
    "voyageNumber": "string or null",
    "containerNumber": "string or null",
    "sealNumber": "string or null",
    "bookingNumber": "string or null",
    "blNumber": "string or null"
  },
  "cargoDetails": {
    "description": "string or null",
    "hsCode": "string or null",
    "quantity": "number or null",
    "unit": "string or null",
    "grossWeight": "string or null",
    "netWeight": "string or null",
    "volume": "string or null",
    "packages": "number or null",
    "packageType": "string or null"
  },
  "financials": {
    "subtotal": "number or null",
    "taxAmount": "number or null",
    "shippingCost": "number or null",
    "insuranceCost": "number or null",
    "totalAmount": "number or null",
    "currency": "string or null",
    "paymentTerms": "string or null",
    "incoterms": "FOB|CIF|EXW|DDP|etc or null"
  },
  "items": [
    {
      "itemNumber": "string",
      "description": "string",
      "quantity": "number",
      "unitPrice": "number",
      "totalPrice": "number"
    }
  ],
  "compliance": {
    "customsValue": "number or null",
    "exportLicense": "string or null",
    "certificateNumber": "string or null",
    "inspectionDate": "string or null"
  },
  "validation": {
    "isComplete": boolean,
    "missingCriticalFields": ["array of missing fields"],
    "complianceScore": 0.0-1.0,
    "readyForBooking": boolean
  },
  "extractedText": "key information summary"
}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, just pure JSON.`;

    console.log('[Invoice AI] Calling OpenAI API...');
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { 
          role: 'system', 
          content: 'You are an expert in shipping documents and invoice processing. Extract ALL available information accurately. Return ONLY valid JSON, no markdown formatting.' 
        },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 3000,
    });

    const responseContent = response.choices[0].message.content;
    
    let jsonString = responseContent.trim();
    if (jsonString.startsWith('```')) {
      jsonString = jsonString.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    }
    
    const analysis = JSON.parse(jsonString);
    console.log('[Invoice AI] ✓ Analysis completed successfully');
    
    return analysis;
    
  } catch (error) {
    console.error('[Invoice AI] ✗ CRITICAL ERROR:', error.message);
    
    return {
      documentType: 'Unknown',
      confidence: 0.2,
      invoiceDetails: {},
      seller: {},
      buyer: {},
      shipmentDetails: {},
      cargoDetails: {},
      financials: {},
      items: [],
      compliance: {},
      validation: {
        isComplete: false,
        missingCriticalFields: [`Analysis failed: ${error.message}`],
        complianceScore: 0.0,
        readyForBooking: false
      },
      extractedText: `Analysis failed: ${error.message}`,
      _error: {
        message: error.message,
        timestamp: new Date().toISOString()
      }
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

      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      console.log(`[PDF Worker] Loaded ${docs.length} pages`);

      const fullText = docs.map(doc => doc.pageContent).join('\n');
      console.log(`[PDF Worker] Running AI analysis...`);
      const aiAnalysis = await analyzeDocumentWithAI(fullText, filename);
      console.log(`[PDF Worker] Detected: ${aiAnalysis.documentType} (confidence: ${aiAnalysis.confidence})`);

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
        separators: ['\n\n', '\n', '. ', ' ', ''],
      });

      const splitDocs = await splitter.splitDocuments(docs);
      console.log(`[PDF Worker] Split into ${splitDocs.length} chunks`);

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
            documentType: aiAnalysis.documentType,
            language: aiAnalysis.language,
            confidence: aiAnalysis.confidence,
            summary: aiAnalysis.summary,
            topics: aiAnalysis.topics,
            sentiment: aiAnalysis.sentiment,
            entities: aiAnalysis.keyEntities,
            fileSize,
            processingVersion: '5.3.0-supabase-only'
          },
        });
      });

      const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-3-small',
        apiKey: process.env.OPENAI_API_KEY,
        batchSize: 100,
        stripNewLines: true,
      });

      console.log(`[PDF Worker] Creating vector store: ${collectionName}`);

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

      // ========== UPDATE SUPABASE DOCUMENT RECORD ==========
      try {
        const { error: updateError } = await supabase
          .from('documents')
          .update({
            total_pages: docs.length,
            total_chunks: processedChunks,
            file_size: fileSize,
            document_type: aiAnalysis.documentType,
            language: aiAnalysis.language,
            confidence: aiAnalysis.confidence,
            summary: aiAnalysis.summary,
            topics: aiAnalysis.topics,
            sentiment: aiAnalysis.sentiment,
            key_entities: aiAnalysis.keyEntities,
            processed: true,
            processed_at: new Date().toISOString(),
            processing_version: '5.3.0-supabase-only'
          })
          .eq('document_id', documentId);

        if (updateError) {
          console.warn(`[PDF Worker] Supabase update failed:`, updateError.message);
        } else {
          console.log(`[PDF Worker] ✓ Updated Supabase record: ${documentId}`);
        }
      } catch (supabaseError) {
        console.warn(`[PDF Worker] Supabase update error:`, supabaseError.message);
      }

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
        version: '5.3.0-supabase-only'
      };

    } catch (err) {
      console.error(`[PDF Worker] Job ${job.id} failed:`, err);
      
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
    connection: redisConnection,
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

// ========== INVOICE PROCESSING WORKER ==========

const invoiceWorker = new Worker(
  'invoice-upload-queue',
  async (job) => {
    const startTime = Date.now();
    
    try {
      console.log(`\n========================================`);
      console.log(`[Invoice Worker] Processing job ${job.id}`);
      console.log(`========================================`);
      
      const data = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;
      const { path: filePath, filename, invoiceId, userId, sessionId, bookingId } = data;

      console.log(`[Invoice Worker] File: ${filename}`);
      console.log(`[Invoice Worker] Invoice ID: ${invoiceId}`);

      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      const fileSize = fs.statSync(filePath).size;
      console.log(`[Invoice Worker] File size: ${(fileSize / 1024).toFixed(2)} KB`);

      console.log(`[Invoice Worker] Loading PDF...`);
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      console.log(`[Invoice Worker] ✓ Loaded ${docs.length} pages`);

      if (docs.length === 0) {
        throw new Error('PDF has no pages or could not be read');
      }

      const fullContent = docs.map(doc => doc.pageContent).join('\n\n');
      console.log(`[Invoice Worker] Extracted text length: ${fullContent.length} characters`);
      
      if (fullContent.length < 50) {
        throw new Error('PDF content is empty or unreadable');
      }

      console.log(`[Invoice Worker] Starting AI analysis...`);
      const analysis = await analyzeInvoiceWithAI(fullContent, filename);
      
      console.log(`\n[Invoice Worker] ========== ANALYSIS RESULTS ==========`);
      console.log(`[Invoice Worker] Document Type: ${analysis.documentType}`);
      console.log(`[Invoice Worker] Confidence: ${analysis.confidence}`);
      console.log(`[Invoice Worker] Ready for Booking: ${analysis.validation.readyForBooking}`);
      console.log(`[Invoice Worker] ======================================\n`);

      console.log(`[Invoice Worker] Updating Supabase...`);
      const { data: updateData, error: updateError } = await supabase
        .from('invoices')
        .update({
          processed: true,
          document_type: analysis.documentType,
          extracted_data: analysis,
          total_pages: docs.length,
          processed_at: new Date().toISOString()
        })
        .eq('invoice_id', invoiceId)
        .select()
        .single();

      if (updateError) {
        throw updateError;
      }
      
      console.log(`[Invoice Worker] ✓ Supabase updated successfully`);

      try {
        console.log(`[Invoice Worker] Creating vector embeddings...`);
        const embeddings = new OpenAIEmbeddings({
          model: 'text-embedding-3-small',
          apiKey: process.env.OPENAI_API_KEY,
        });

        const collectionName = `invoices_${userId}`;
        
        const invoiceDoc = new Document({
          pageContent: fullContent,
          metadata: {
            source: filename,
            invoiceId,
            sessionId,
            bookingId,
            userId,
            type: 'invoice',
            documentType: analysis.documentType,
            analysis: analysis,
            processedAt: new Date().toISOString(),
            fileSize,
            version: '5.3.0-supabase-only'
          },
        });

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

        await vectorStore.addDocuments([invoiceDoc]);
        console.log(`[Invoice Worker] ✓ Vector embeddings created in Qdrant`);
      } catch (vectorError) {
        console.warn(`[Invoice Worker] Vector storage failed:`, vectorError.message);
      }

      try {
        fs.unlinkSync(filePath);
        console.log(`[Invoice Worker] ✓ Cleaned up file`);
      } catch (err) {
        console.warn(`[Invoice Worker] Could not delete file: ${err.message}`);
      }

      const processingTime = Date.now() - startTime;
      console.log(`\n[Invoice Worker] ✓✓✓ COMPLETED IN ${processingTime}ms ✓✓✓\n`);

      return {
        success: true,
        filename,
        invoiceId,
        sessionId,
        bookingId,
        analysis,
        collectionName: `invoices_${userId}`,
        processingTime,
        version: '5.3.0-supabase-only'
      };

    } catch (err) {
      console.error(`\n[Invoice Worker] ✗✗✗ JOB FAILED ✗✗✗`);
      console.error(`[Invoice Worker] Error:`, err.message);
      
      try {
        const data = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;
        await supabase
          .from('invoices')
          .update({
            processed: false,
            extracted_data: { 
              error: err.message,
              timestamp: new Date().toISOString()
            }
          })
          .eq('invoice_id', data.invoiceId);
      } catch (dbErr) {
        console.error(`[Invoice Worker] Could not update database:`, dbErr.message);
      }
      
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
    concurrency: 1,
    connection: redisConnection,
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
  console.log(`[PDF Worker] ✓ Job ${job.id} completed: ${result.filename}`);
});

pdfWorker.on('failed', (job, err) => {
  console.error(`[PDF Worker] ✗ Job ${job?.id} failed: ${err.message}`);
});

invoiceWorker.on('completed', (job, result) => {
  console.log(`[Invoice Worker] ✓ Job ${job.id} completed: ${result.filename}`);
});

invoiceWorker.on('failed', (job, err) => {
  console.error(`[Invoice Worker] ✗ Job ${job?.id} failed: ${err.message}`);
});

// ========== HEALTH MONITORING ==========

setInterval(() => {
  const pdfStatus = pdfWorker.isRunning() ? '✓ Running' : '✗ Stopped';
  const invoiceStatus = invoiceWorker.isRunning() ? '✓ Running' : '✗ Stopped';
  console.log(`[Workers] PDF: ${pdfStatus} | Invoice: ${invoiceStatus}`);
}, 60000);

// ========== HEALTH CHECK API WITH CORS ==========

const workerApp = express();
const WORKER_PORT = process.env.WORKER_PORT || 8001;

workerApp.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

workerApp.use(express.json());

workerApp.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    workers: [
      {
        name: 'PDF Processing',
        queue: 'file-upload-queue',
        status: pdfWorker.isRunning() ? 'running' : 'stopped',
        concurrency: 2
      },
      {
        name: 'Invoice Processing',
        queue: 'invoice-upload-queue', 
        status: invoiceWorker.isRunning() ? 'running' : 'stopped',
        concurrency: 1
      }
    ],
    storage: 'Supabase (Qdrant for vectors)',
    cache: 'None (direct Supabase queries)',
    deployment: 'Render Free Tier',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    version: '5.3.0-supabase-only'
  });
});

// ========== SUPABASE DATA RETRIEVAL ENDPOINTS ==========

workerApp.get('/api/invoice/:invoiceId', async (req, res) => {
  try {
    const { invoiceId } = req.params;
    const { data, error } = await supabase
      .from('invoices')
      .select('*')
      .eq('invoice_id', invoiceId)
      .single();
    
    if (error) throw error;
    if (!data) return res.status(404).json({ error: 'Invoice not found' });
    
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

workerApp.get('/api/document/:documentId', async (req, res) => {
  try {
    const { documentId } = req.params;
    const { data, error } = await supabase
      .from('documents')
      .select('*')
      .eq('document_id', documentId)
      .single();
    
    if (error) throw error;
    if (!data) return res.status(404).json({ error: 'Document not found' });
    
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

workerApp.get('/api/user/:userId/invoices', async (req, res) => {
  try {
    const { userId } = req.params;
    const { data, error } = await supabase
      .from('invoices')
      .select('invoice_id, filename, document_type, uploaded_at, processed, extracted_data')
      .eq('user_id', userId)
      .order('uploaded_at', { ascending: false });
    
    if (error) throw error;
    
    res.json({ 
      userId, 
      invoices: data || [],
      count: data?.length || 0
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

workerApp.get('/api/user/:userId/documents', async (req, res) => {
  try {
    const { userId } = req.params;
    const { data, error } = await supabase
      .from('documents')
      .select('document_id, filename, document_type, uploaded_at, processed, total_pages, total_chunks')
      .eq('user_id', userId)
      .order('uploaded_at', { ascending: false });
    
    if (error) throw error;
    
    res.json({ 
      userId, 
      documents: data || [],
      count: data?.length || 0
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

workerApp.listen(WORKER_PORT, () => {
  console.log('========================================');
  console.log('FreightChat Pro Workers Started');
  console.log('========================================');
  console.log(`Version: 5.3.0-supabase-only`);
  console.log(`Health Check: http://localhost:${WORKER_PORT}/health`);
  console.log(`Storage: Supabase (single source of truth)`);
  console.log(`Vectors: Qdrant`);
  console.log(`Queue: BullMQ (Redis for jobs only)`);
  console.log(`Deployment: Render Free Tier Ready`);
  console.log(`CORS: Enabled for localhost:3000, localhost:3001`);
  console.log('========================================');
  console.log('Active Workers:');
  console.log('  ✓ PDF Processing (2 concurrent)');
  console.log('  ✓ Invoice Processing (1 concurrent)');
  console.log('========================================');
  console.log('API Endpoints:');
  console.log('  GET /api/invoice/:invoiceId');
  console.log('  GET /api/document/:documentId');
  console.log('  GET /api/user/:userId/invoices');
  console.log('  GET /api/user/:userId/documents');
  console.log('========================================');
  console.log('Ready to process jobs...');
});

// ========== GRACEFUL SHUTDOWN ==========

async function gracefulShutdown() {
  console.log('\n[Shutdown] Gracefully shutting down workers...');
  
  try {
    await Promise.all([
      pdfWorker.close(),
      invoiceWorker.close()
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

console.log('Workers initialized and ready!');