import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { Queue } from 'bullmq';
import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantVectorStore } from '@langchain/qdrant';
import fs from 'fs';
import dotenv from 'dotenv';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';
import { createClient } from '@supabase/supabase-js';
import { MemorySaver } from '@langchain/langgraph';
import { createShippingAgent, ShippingAgentExecutor } from './workflow.js';
import { redisConnection } from './redis-config.js';
import cloudinary from './cloudinary-config.js';
import { CloudinaryStorage } from 'multer-storage-cloudinary';

dotenv.config();

console.log('🚀 Starting FreightChat Pro API...');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

console.log('✓ Supabase client initialized');

// Initialize LangGraph checkpointer
const checkpointer = new MemorySaver();
let shippingAgent = null;

console.log('✓ MemorySaver checkpointer created');

// Initialize agent on startup
console.log('🤖 Initializing shipping agent...');
createShippingAgent(checkpointer)
  .then((agent) => {
    shippingAgent = new ShippingAgentExecutor(agent, checkpointer);
    console.log('✓ Shipping Agent initialized successfully');
  })
  .catch((error) => {
    console.error('✗ Failed to initialize agent:', error);
    console.error('Stack:', error.stack);
  });

// BullMQ Queues with Upstash Redis connection
const queue = new Queue('file-upload-queue', {
  connection: redisConnection
});

const invoiceQueue = new Queue('invoice-upload-queue', {
  connection: redisConnection
});

console.log('✓ BullMQ queues initialized with Upstash Redis');

// FIXED: Cloudinary storage configuration - let Cloudinary auto-generate public_id
const cloudinaryStorage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: 'freightchat-documents',
    resource_type: 'raw',
    allowed_formats: ['pdf'],
    use_filename: true,
    unique_filename: true,
  },
});

// Multer upload configuration with Cloudinary
const upload = multer({ 
  storage: cloudinaryStorage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files allowed'), false);
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

const app = express();

// CORS configuration for deployment
const corsOptions = {
  origin: process.env.FRONTEND_URL 
    ? [process.env.FRONTEND_URL, 'http://localhost:3000']
    : '*',
  credentials: true
};

app.use(cors(corsOptions));
app.use(express.json());

console.log('✓ Express middleware configured');

// Database helpers
async function createUser(userData) {
  const { data, error } = await supabase
    .from('users')
    .insert([{
      user_id: userData.userId,
      name: userData.name,
      email: userData.email,
      created_at: new Date().toISOString(),
      last_accessed: new Date().toISOString(),
      is_active: true
    }])
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function getUserById(userId) {
  const { data, error } = await supabase
    .from('users')
    .select('*')
    .eq('user_id', userId)
    .single();
  if (error && error.code !== 'PGRST116') throw error;
  return data;
}

async function updateUserLastAccessed(userId) {
  const { error } = await supabase
    .from('users')
    .update({ last_accessed: new Date().toISOString() })
    .eq('user_id', userId);
  if (error) throw error;
}

async function createDocument(docData) {
  const { data, error } = await supabase
    .from('documents')
    .insert([{
      document_id: docData.documentId,
      user_id: docData.userId,
      filename: docData.filename,
      collection_name: docData.collectionName,
      strategy: docData.strategy,
      cloudinary_url: docData.cloudinaryUrl,
      cloudinary_public_id: docData.cloudinaryPublicId,
      uploaded_at: new Date().toISOString()
    }])
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function getUserDocuments(userId) {
  const { data, error } = await supabase
    .from('documents')
    .select('*')
    .eq('user_id', userId)
    .order('uploaded_at', { ascending: false });
  if (error) throw error;
  return data || [];
}

async function saveShippingQuote(sessionId, quoteData, userId) {
  const { data, error } = await supabase
    .from('shipping_quotes')
    .insert([{
      session_id: sessionId,
      user_id: userId,
      quote_data: quoteData,
      created_at: new Date().toISOString()
    }])
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function createShipmentTracking(bookingData) {
  const { data, error } = await supabase
    .from('shipment_tracking')
    .insert([{
      tracking_number: bookingData.trackingNumber,
      booking_id: bookingData.bookingId,
      user_id: bookingData.userId,
      session_id: bookingData.sessionId,
      carrier_id: bookingData.carrierId,
      service_level: bookingData.serviceLevel,
      origin: bookingData.origin,
      destination: bookingData.destination,
      status: 'pickup_scheduled',
      estimated_delivery: bookingData.estimatedDelivery,
      tracking_events: [],
      created_at: new Date().toISOString()
    }])
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function createInvoiceRecord(invoiceData) {
  const { data, error } = await supabase
    .from('invoices')
    .insert([{
      invoice_id: invoiceData.invoiceId,
      user_id: invoiceData.userId,
      session_id: invoiceData.sessionId,
      booking_id: invoiceData.bookingId,
      filename: invoiceData.filename,
      cloudinary_url: invoiceData.cloudinaryUrl,
      cloudinary_public_id: invoiceData.cloudinaryPublicId,
      file_size: invoiceData.fileSize,
      document_type: invoiceData.documentType || 'invoice',
      extracted_data: invoiceData.extractedData || {},
      uploaded_at: new Date().toISOString(),
      processed: false
    }])
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function getSessionInvoices(sessionId) {
  const { data, error } = await supabase
    .from('invoices')
    .select('*')
    .eq('session_id', sessionId)
    .order('uploaded_at', { ascending: false });
  if (error) throw error;
  return data || [];
}

async function getShipmentTracking(trackingNumber, userId = null) {
  let query = supabase
    .from('shipment_tracking')
    .select('*')
    .eq('tracking_number', trackingNumber);
  if (userId) query = query.eq('user_id', userId);
  const { data, error } = await query.single();
  if (error && error.code !== 'PGRST116') throw error;
  return data;
}

async function getUserShipments(userId) {
  const { data, error } = await supabase
    .from('shipment_tracking')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });
  if (error) throw error;
  return data || [];
}

function isValidUserId(userId) {
  return /^[a-zA-Z0-9_-]{3,50}$/.test(userId);
}

function generateUserToken(userId) {
  return jwt.sign(
    { userId, createdAt: Date.now(), type: 'auth' },
    process.env.JWT_SECRET || 'your-secret-key',
    { expiresIn: '30d' }
  );
}

async function verifyUserToken(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1] || req.query.token;
  if (!token) {
    return res.status(401).json({ 
      error: 'Authentication required',
      requiresAuth: true
    });
  }
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    const user = await getUserById(decoded.userId);
    if (!user) {
      return res.status(401).json({ 
        error: 'User not found',
        requiresAuth: true
      });
    }
    req.userId = decoded.userId;
    await updateUserLastAccessed(decoded.userId);
  } catch (error) {
    if (error.name === 'JsonWebTokenError' || error.name === 'TokenExpiredError') {
      return res.status(401).json({ 
        error: 'Invalid or expired token',
        requiresAuth: true
      });
    }
    return res.status(500).json({ error: 'Authentication failed' });
  }
  next();
}

// Health check
app.get('/', async (req, res) => {
  try {
    const { count: userCount } = await supabase
      .from('users')
      .select('*', { count: 'exact', head: true });
    
    const { count: docCount } = await supabase
      .from('documents')
      .select('*', { count: 'exact', head: true });

    const { count: trackingCount } = await supabase
      .from('shipment_tracking')
      .select('*', { count: 'exact', head: true });

    const { count: invoiceCount } = await supabase
      .from('invoices')
      .select('*', { count: 'exact', head: true });

    return res.json({ 
      status: 'FreightChat Pro API - Cloudinary Integration',
      registeredUsers: userCount || 0,
      totalDocuments: docCount || 0,
      activeShipments: trackingCount || 0,
      invoicesProcessed: invoiceCount || 0,
      database: 'Supabase Connected',
      redis: 'Upstash Connected',
      storage: 'Cloudinary Active',
      agentStatus: shippingAgent ? '✓ Active' : '⚠ Initializing',
      architecture: 'LangGraph Multi-Agent',
      deployment: 'Render Free Tier Ready',
      features: ['PDF Chat', 'AI Shipping Agent', 'Real-time Tracking', 'Invoice Processing'],
      version: '5.4.1-cloudinary-fixed'
    });
  } catch (error) {
    return res.status(500).json({
      status: 'Error',
      error: error.message
    });
  }
});

// Auth endpoints
app.post('/auth/register', async (req, res) => {
  try {
    const { userId, name, email } = req.body;
    if (!userId) {
      return res.status(400).json({ error: 'userId is required' });
    }
    if (!isValidUserId(userId)) {
      return res.status(400).json({ error: 'Invalid userId format' });
    }
    const existingUser = await getUserById(userId);
    if (existingUser) {
      return res.status(409).json({ error: 'User ID already exists' });
    }
    const userInfo = await createUser({ userId, name: name || userId, email: email || null });
    const token = generateUserToken(userId);
    return res.status(201).json({
      message: 'User created successfully',
      user: {
        userId: userInfo.user_id,
        name: userInfo.name,
        email: userInfo.email,
        createdAt: userInfo.created_at
      },
      token,
      expiresIn: '30 days'
    });
  } catch (error) {
    console.error('Error creating user:', error);
    return res.status(500).json({ error: 'Failed to create user' });
  }
});

app.post('/auth/login', async (req, res) => {
  try {
    const { userId } = req.body;
    if (!userId) {
      return res.status(400).json({ error: 'userId is required' });
    }
    const userInfo = await getUserById(userId);
    if (!userInfo) {
      return res.status(404).json({ error: 'User not found' });
    }
    if (!userInfo.is_active) {
      return res.status(403).json({ error: 'User account is inactive' });
    }
    await updateUserLastAccessed(userId);
    const token = generateUserToken(userId);
    return res.json({
      message: 'Login successful',
      user: {
        userId: userInfo.user_id,
        name: userInfo.name,
        email: userInfo.email,
        lastAccessed: new Date().toISOString()
      },
      token,
      expiresIn: '30 days'
    });
  } catch (error) {
    console.error('Error during login:', error);
    return res.status(500).json({ error: 'Login failed' });
  }
});

app.get('/auth/profile', verifyUserToken, async (req, res) => {
  try {
    const userInfo = await getUserById(req.userId);
    const documents = await getUserDocuments(req.userId);
    const shipments = await getUserShipments(req.userId);
    return res.json({
      user: {
        userId: userInfo.user_id,
        name: userInfo.name,
        email: userInfo.email,
        createdAt: userInfo.created_at,
        lastAccessed: userInfo.last_accessed
      },
      documents: documents,
      documentCount: documents.length,
      shipmentsCount: shipments.length,
      activeShipments: shipments.filter(s => !['delivered', 'returned'].includes(s.status)).length
    });
  } catch (error) {
    console.error('Error fetching profile:', error);
    return res.status(500).json({ error: 'Failed to fetch profile' });
  }
});

// PDF upload with Cloudinary
app.post('/upload/pdf', verifyUserToken, upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    const userId = req.userId;
    const documentId = crypto.randomBytes(16).toString('hex');
    const strategy = req.body.strategy || 'user';
    
    // Cloudinary info from multer-storage-cloudinary
    const cloudinaryUrl = req.file.path; // Full Cloudinary URL
    const cloudinaryPublicId = req.file.filename; // Cloudinary public_id
    
    let collectionName;
    switch (strategy) {
      case 'document':
        collectionName = `doc_${documentId}`;
        break;
      case 'shared':
        collectionName = process.env.QDRANT_COLLECTION || 'shared_documents';
        break;
      case 'user':
      default:
        collectionName = `user_${userId}`;
        break;
    }

    await createDocument({ 
      documentId, 
      userId, 
      filename: req.file.originalname, 
      collectionName, 
      strategy,
      cloudinaryUrl,
      cloudinaryPublicId
    });

    await queue.add('file-ready', {
      filename: req.file.originalname,
      cloudinaryUrl,
      cloudinaryPublicId,
      userId,
      documentId,
      collectionName,
      strategy,
      metadata: {
        uploadedBy: userId,
        uploadedAt: new Date().toISOString(),
      }
    });

    return res.json({ 
      message: 'PDF uploaded to Cloudinary and queued for processing',
      filename: req.file.originalname,
      documentId,
      userId,
      collectionName,
      strategy,
      cloudinaryUrl
    });
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return res.status(500).json({ error: 'Failed to upload PDF' });
  }
});

// Invoice upload during agent conversation with Cloudinary
app.post('/agent/shipping/upload-invoice', verifyUserToken, upload.single('invoice'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No invoice file uploaded' });
    }

    const { threadId, bookingId } = req.body;
    
    if (!threadId) {
      return res.status(400).json({ error: 'threadId is required' });
    }

    const userId = req.userId;
    const invoiceId = crypto.randomBytes(16).toString('hex');
    
    // Cloudinary info
    const cloudinaryUrl = req.file.path;
    const cloudinaryPublicId = req.file.filename;
    const fileSize = req.file.size;

    const invoiceRecord = await createInvoiceRecord({
      invoiceId,
      userId,
      sessionId: threadId,
      bookingId: bookingId || null,
      filename: req.file.originalname,
      cloudinaryUrl,
      cloudinaryPublicId,
      fileSize
    });

    await invoiceQueue.add('process-invoice', {
      invoiceId,
      filename: req.file.originalname,
      cloudinaryUrl,
      cloudinaryPublicId,
      userId,
      sessionId: threadId,
      bookingId: bookingId || null,
      uploadedAt: new Date().toISOString()
    });

    if (shippingAgent) {
      const config = { configurable: { thread_id: threadId } };
      const snapshot = await checkpointer.get(config);
      
      if (snapshot) {
        const currentState = snapshot.channel_values;
        
        if (!currentState.shipmentData.invoices) {
          currentState.shipmentData.invoices = [];
        }
        
        currentState.shipmentData.invoices.push({
          invoiceId,
          filename: req.file.originalname,
          uploadedAt: new Date().toISOString(),
          processed: false
        });

        currentState.messages.push({
          role: 'system',
          content: `Invoice uploaded: ${req.file.originalname}`,
          timestamp: new Date().toISOString()
        });
      }
    }

    return res.json({
      success: true,
      message: 'Invoice uploaded to Cloudinary and queued for AI analysis',
      invoiceId,
      filename: req.file.originalname,
      fileSize,
      sessionId: threadId,
      cloudinaryUrl,
      processing: 'AI analysis in progress'
    });

  } catch (error) {
    console.error('Error uploading invoice:', error);
    return res.status(500).json({ error: 'Failed to upload invoice' });
  }
});

// Get invoices for a session
app.get('/agent/shipping/invoices/:threadId', verifyUserToken, async (req, res) => {
  try {
    const { threadId } = req.params;
    const invoices = await getSessionInvoices(threadId);
    
    return res.json({
      success: true,
      threadId,
      invoices: invoices.map(inv => ({
        invoiceId: inv.invoice_id,
        filename: inv.filename,
        uploadedAt: inv.uploaded_at,
        processed: inv.processed,
        extractedData: inv.extracted_data,
        documentType: inv.document_type,
        cloudinaryUrl: inv.cloudinary_url
      })),
      count: invoices.length
    });
  } catch (error) {
    console.error('Error fetching invoices:', error);
    return res.status(500).json({ error: 'Failed to fetch invoices' });
  }
});

// Document chat endpoint
app.get('/chat/documents', verifyUserToken, async (req, res) => {
  try {
    const userQuery = req.query.message;
    const strategy = req.query.strategy || 'user';
    
    if (!userQuery) {
      return res.status(400).json({ error: 'Message query parameter is required' });
    }

    const userId = req.userId;
    const userDocs = await getUserDocuments(userId);
    
    if (userDocs.length === 0) {
      return res.json({
        message: "No documents uploaded yet. Upload a PDF to chat with your documents.",
        docsFound: 0
      });
    }

    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-small',
      apiKey: process.env.OPENAI_API_KEY,
    });

    let relevantDocs = [];
    if (strategy === 'user') {
      const collectionName = `user_${userId}`;
      try {
        const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
          url: process.env.QDRANT_URL,
          apiKey: process.env.QDRANT_API_KEY,
          collectionName,
          checkCompatibility: false,
        });
        const retriever = vectorStore.asRetriever({ k: 5 });
        relevantDocs = await retriever.invoke(userQuery);
      } catch (error) {
        console.log(`Collection ${collectionName} not found`);
      }
    }

    if (relevantDocs.length === 0) {
      return res.json({
        message: "No relevant information found in your documents.",
        docsFound: 0
      });
    }

    if (!shippingAgent) {
      return res.status(503).json({ error: 'Agent not ready' });
    }

    const threadId = `doc_chat_${Date.now()}`;
    const context = relevantDocs.map((doc, i) => 
      `Document ${i + 1}:\n${doc.pageContent}`
    ).join('\n\n');

    const initialState = {
      messages: [{
        role: 'user',
        content: `Based on these documents, answer: ${userQuery}\n\nContext:\n${context}`,
        timestamp: new Date().toISOString()
      }],
      userId,
      threadId,
      shipmentData: {},
      currentPhase: 'document_analysis',
      completed: false,
      quote: null,
      output: null,
      nextAction: 'respond'
    };

    const config = { configurable: { thread_id: threadId } };
    const result = await shippingAgent.invoke(initialState, config);

    return res.json({
      message: result.output,
      query: userQuery,
      docsFound: relevantDocs.length,
      userId,
      mode: 'document_chat'
    });

  } catch (error) {
    console.error('Document chat error:', error);
    return res.status(500).json({ error: 'Failed to process document chat' });
  }
});

// Agent endpoints
app.post('/agent/shipping/start', verifyUserToken, async (req, res) => {
  try {
    if (!shippingAgent) {
      return res.status(503).json({ 
        error: 'Agent not initialized yet', 
        retry: true 
      });
    }

    const threadId = `thread_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
    
    const initialState = {
      messages: [],
      userId: req.userId,
      threadId,
      shipmentData: {},
      currentPhase: 'greeting',
      completed: false,
      quote: null,
      output: null,
      nextAction: null
    };

    const config = {
      configurable: { thread_id: threadId }
    };

    const result = await shippingAgent.invoke(initialState, config);

    return res.json({
      success: true,
      threadId,
      message: result.output || "Welcome! I'm your AI shipping agent. Let's get started with your shipment. Where are you shipping from and to?",
      currentPhase: result.currentPhase,
      completed: result.completed,
      architecture: 'LangGraph Agent',
      features: {
        invoiceUpload: true,
        uploadEndpoint: '/agent/shipping/upload-invoice',
        storage: 'Cloudinary'
      }
    });

  } catch (error) {
    console.error('Agent start error:', error);
    return res.status(500).json({ 
      success: false, 
      error: 'Failed to start agent',
      details: error.message 
    });
  }
});

app.post('/agent/shipping/message', verifyUserToken, async (req, res) => {
  try {
    const { threadId, message } = req.body;
    
    if (!threadId || !message) {
      return res.status(400).json({ error: 'threadId and message required' });
    }

    if (!shippingAgent) {
      return res.status(503).json({ error: 'Agent not initialized' });
    }

    const config = {
      configurable: { thread_id: threadId }
    };

    const snapshot = await checkpointer.get(config);
    
    if (!snapshot) {
      return res.status(404).json({ error: 'Session not found or expired' });
    }

    const currentState = snapshot.channel_values;

    currentState.messages.push({
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    });

    const result = await shippingAgent.invoke(currentState, config);

    if (result.completed && result.quote) {
      try {
        await saveShippingQuote(threadId, result.quote, req.userId);
      } catch (error) {
        console.error('Failed to save quote:', error);
      }
    }

    return res.json({
      success: true,
      threadId,
      message: result.output,
      currentPhase: result.currentPhase,
      shipmentData: result.shipmentData,
      quote: result.quote,
      completed: result.completed,
      nextAction: result.nextAction,
      invoices: result.shipmentData.invoices || []
    });

  } catch (error) {
    console.error('Agent message error:', error);
    return res.status(500).json({ 
      success: false, 
      error: 'Failed to process message',
      details: error.message 
    });
  }
});

app.post('/agent/shipping/book', verifyUserToken, async (req, res) => {
  try {
    const { threadId, carrierId, serviceLevel } = req.body;
    
    if (!threadId || !carrierId) {
      return res.status(400).json({ error: 'threadId and carrierId required' });
    }

    const config = {
      configurable: { thread_id: threadId }
    };

    const snapshot = await checkpointer.get(config);
    if (!snapshot) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const state = snapshot.channel_values;

    const bookingId = `BK${Date.now()}`;
    const trackingNumber = `FCP${crypto.randomBytes(5).toString('hex').toUpperCase()}`;
    const estimatedDelivery = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

    await createShipmentTracking({
      trackingNumber,
      bookingId,
      userId: req.userId,
      sessionId: threadId,
      carrierId,
      serviceLevel: serviceLevel || state.shipmentData?.serviceLevel || 'Standard',
      origin: state.shipmentData?.origin || 'Origin',
      destination: state.shipmentData?.destination || 'Destination',
      estimatedDelivery: estimatedDelivery.toISOString()
    });

    // Link invoices to booking
    if (state.shipmentData?.invoices?.length > 0) {
      for (const invoice of state.shipmentData.invoices) {
        await supabase
          .from('invoices')
          .update({ booking_id: bookingId })
          .eq('invoice_id', invoice.invoiceId);
      }
    }

    return res.json({
      success: true,
      message: 'Shipment booked successfully!',
      bookingId,
      trackingNumber,
      carrierId,
      estimatedDelivery: estimatedDelivery.toISOString(),
      linkedInvoices: state.shipmentData?.invoices?.length || 0
    });

  } catch (error) {
    console.error('Booking error:', error);
    return res.status(500).json({ error: 'Failed to book shipment' });
  }
});

// Tracking endpoints
app.get('/track/:trackingNumber', async (req, res) => {
  try {
    const tracking = await getShipmentTracking(req.params.trackingNumber);
    if (!tracking) {
      return res.status(404).json({ error: 'Tracking number not found' });
    }
    return res.json({
      trackingNumber: tracking.tracking_number,
      status: tracking.status,
      currentLocation: tracking.current_location,
      origin: tracking.origin,
      destination: tracking.destination,
      carrier: tracking.carrier_id,
      estimatedDelivery: tracking.estimated_delivery,
      trackingEvents: tracking.tracking_events || []
    });
  } catch (error) {
    console.error('Tracking error:', error);
    return res.status(500).json({ error: 'Failed to fetch tracking' });
  }
});

app.get('/shipments', verifyUserToken, async (req, res) => {
  try {
    const shipments = await getUserShipments(req.userId);
    return res.json({
      total: shipments.length,
      activeShipments: shipments.filter(s => !['delivered', 'returned'].includes(s.status)),
      recentShipments: shipments.slice(0, 10)
    });
  } catch (error) {
    console.error('Shipments error:', error);
    return res.status(500).json({ error: 'Failed to fetch shipments' });
  }
});

// Error handler
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Max 10MB.' });
    }
  }
  console.error('Error handler:', error);
  return res.status(500).json({ error: error.message });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log('========================================');
  console.log(`✓ FreightChat Pro API Running`);
  console.log(`📍 Port: ${PORT}`);
  console.log(`🌐 Health: http://localhost:${PORT}/`);
  console.log(`📦 Version: 5.4.1 - Cloudinary Signature Fixed`);
  console.log(`🤖 Agent Status: ${shippingAgent ? '✓ Ready' : '⚠ Initializing...'}`);
  console.log(`📄 Invoice Processing: ✓ Active`);
  console.log(`☁️  Storage: Cloudinary Connected`);
  console.log(`💾 Redis: Upstash Connected`);
  console.log(`🚀 Deployment: Render Free Tier Ready`);
  console.log('========================================');
});