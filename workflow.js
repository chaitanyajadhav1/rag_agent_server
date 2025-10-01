import { StateGraph, END, START } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0.7,
  apiKey: process.env.OPENAI_API_KEY,
});

const ShippingStateSchema = {
  messages: [],
  userId: null,
  threadId: null,
  shipmentData: {},
  currentPhase: 'greeting',
  completed: false,
  quote: null,
  output: null,
  nextAction: null
};

// Single agent that handles everything
async function shippingAgentNode(state) {
  // If no messages, start with greeting
  if (state.messages.length === 0) {
    const greetingMessage = {
      role: 'assistant',
      content: "Hello! I'm your AI shipping agent. I'll help you get the best freight quotes and manage your shipment.\n\nTo start, could you tell me:\n1. Where are you shipping from?\n2. Where are you shipping to?\n\n(Example: From Mumbai, India to New York, USA)\n\nYou can also upload invoices or shipping documents at any time during our conversation.",
      timestamp: new Date().toISOString()
    };
    
    return {
      ...state,
      messages: [greetingMessage],
      output: greetingMessage.content,
      currentPhase: 'route_collection',
      completed: false
    };
  }

  // Get last message
  const lastMessage = state.messages[state.messages.length - 1];
  
  // If last message was from assistant, we're waiting for user input
  if (lastMessage.role === 'assistant') {
    return {
      ...state,
      output: lastMessage.content,
      completed: false
    };
  }

  // Check for system messages about invoice uploads
  const systemMessages = state.messages.filter(m => m.role === 'system');
  const hasNewInvoice = systemMessages.length > 0 && 
    systemMessages[systemMessages.length - 1].content.includes('Invoice uploaded');

  // User has sent a message - process it
  const userMessage = lastMessage.content;
  
  // Build conversation context (last 10 messages for better context)
  const conversationHistory = state.messages.slice(-10).map(m => {
    if (m.role === 'user') return new HumanMessage(m.content);
    if (m.role === 'assistant') return new AIMessage(m.content);
    if (m.role === 'system') return new SystemMessage(m.content);
    return new HumanMessage(m.content);
  });

  // Build context about uploaded invoices
  const invoiceContext = state.shipmentData.invoices && state.shipmentData.invoices.length > 0
    ? `\n\nUPLOADED INVOICES:\n${state.shipmentData.invoices.map(inv => 
        `- ${inv.filename} (${inv.processed ? 'Processed' : 'Processing...'})`
      ).join('\n')}`
    : '';

  // System prompt for the agent
  const systemPrompt = `You are a professional freight shipping assistant helping users book shipments.

Current Phase: ${state.currentPhase}
Collected Data: ${JSON.stringify(state.shipmentData, null, 2)}${invoiceContext}

CRITICAL INSTRUCTIONS:
1. Extract shipping information from user messages and update the shipmentData
2. Guide users through collecting these details (one at a time):
   - Origin location (city, country)
   - Destination location (city, country)
   - Cargo description
   - Weight (in kg)
   - Service level preference (Express/Standard/Economy)
   - Special requirements (optional)
   - Declared value (optional)
   - Contact information (optional)

3. When user uploads an invoice, acknowledge it warmly and explain you'll use it to auto-fill details

4. GENERATE QUOTE when you have AT MINIMUM:
   - Origin
   - Destination
   - Cargo description
   - Weight (approximate is fine)

5. Response format:
   - For normal conversation: Just respond naturally
   - When ready for quote: Start your response with "READY_FOR_QUOTE" on first line, then provide natural response

Be conversational, friendly, and professional. Ask for ONE thing at a time. Don't overwhelm the user.`;

  try {
    const response = await model.invoke([
      new SystemMessage(systemPrompt),
      ...conversationHistory
    ]);

    // Handle response content safely
    const responseText = typeof response.content === 'string' 
      ? response.content 
      : JSON.stringify(response.content);

    const shouldGenerateQuote = responseText.trim().startsWith('READY_FOR_QUOTE');
    
    // Remove the READY_FOR_QUOTE marker from display
    const assistantResponse = shouldGenerateQuote 
      ? responseText.replace(/^READY_FOR_QUOTE\s*/i, '').trim()
      : responseText;

    // Extract data from conversation using structured extraction
    let extractedData = { ...state.shipmentData };
    
    // Use AI to extract structured data
    const extractionPrompt = `Extract shipping information from this user message and conversation context.

User message: "${userMessage}"

Current data: ${JSON.stringify(state.shipmentData, null, 2)}

Return ONLY valid JSON with any found/updated fields (use null for missing fields):
{
  "origin": "city, country or null",
  "destination": "city, country or null", 
  "cargo": "description or null",
  "weight": "number with unit (e.g., '50kg') or null",
  "serviceLevel": "Express|Standard|Economy or null",
  "specialRequirements": "description or null",
  "declaredValue": "amount or null",
  "contactName": "name or null",
  "contactEmail": "email or null",
  "contactPhone": "phone or null"
}`;

    try {
      const extractResponse = await model.invoke([
        new SystemMessage(extractionPrompt),
        new HumanMessage(userMessage)
      ]);

      const extractedText = typeof extractResponse.content === 'string'
        ? extractResponse.content
        : JSON.stringify(extractResponse.content);

      // Try to parse JSON from response
      const jsonMatch = extractedText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        // Only update fields that have non-null values
        Object.keys(parsed).forEach(key => {
          if (parsed[key] !== null && parsed[key] !== '') {
            extractedData[key] = parsed[key];
          }
        });
      }
    } catch (e) {
      console.log('Data extraction failed, keeping existing data:', e.message);
    }

    // Generate quote if ready
    if (shouldGenerateQuote && extractedData.origin && extractedData.destination && extractedData.cargo) {
      const quote = await generateShippingQuote(extractedData);
      
      const quoteMessage = `${assistantResponse}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHIPPING QUOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Top 3 Recommended Carriers:**

1. **${quote.quotes[0].name}**
   Rate: $${quote.quotes[0].rate}
   Transit: ${quote.quotes[0].transitTime}
   Reliability: ${quote.quotes[0].reliability}

2. **${quote.quotes[1].name}**
   Rate: $${quote.quotes[1].rate}
   Transit: ${quote.quotes[1].transitTime}
   Reliability: ${quote.quotes[1].reliability}

3. **${quote.quotes[2].name}**
   Rate: $${quote.quotes[2].rate}
   Transit: ${quote.quotes[2].transitTime}
   Reliability: ${quote.quotes[2].reliability}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shipment Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Route: ${extractedData.origin || 'Origin'} → ${extractedData.destination || 'Destination'}
Weight: ${extractedData.weight || 'Not specified'}
Service: ${extractedData.serviceLevel || 'Standard'}
${extractedData.specialRequirements ? `Special: ${extractedData.specialRequirements}` : ''}
${state.shipmentData.invoices?.length > 0 ? `Invoices: ${state.shipmentData.invoices.length} uploaded` : ''}

Would you like to book one of these carriers?`;

      state.messages.push({
        role: 'assistant',
        content: quoteMessage,
        timestamp: new Date().toISOString()
      });

      return {
        ...state,
        shipmentData: extractedData,
        quote,
        output: quoteMessage,
        currentPhase: 'quote_generated',
        completed: true
      };
    }

    // Continue conversation
    state.messages.push({
      role: 'assistant',
      content: assistantResponse,
      timestamp: new Date().toISOString()
    });

    return {
      ...state,
      shipmentData: extractedData,
      output: assistantResponse,
      currentPhase: determinePhase(extractedData),
      completed: false
    };

  } catch (error) {
    console.error('Agent error:', error);
    const errorMessage = "I apologize, I encountered an error processing your request. Could you please try rephrasing that?";
    
    state.messages.push({
      role: 'assistant',
      content: errorMessage,
      timestamp: new Date().toISOString()
    });

    return {
      ...state,
      output: errorMessage,
      completed: false
    };
  }
}

function determinePhase(data) {
  if (data.quote) return 'quote_generated';
  if (data.origin && data.destination && data.cargo && data.weight) return 'ready_for_quote';
  if (data.cargo && data.weight) return 'service_selection';
  if (data.origin && data.destination) return 'cargo_collection';
  if (data.origin || data.destination) return 'route_collection';
  return 'route_collection';
}

export async function createShippingAgent(checkpointer) {
  const workflow = new StateGraph({
    channels: ShippingStateSchema
  });

  workflow.addNode('agent', shippingAgentNode);
  workflow.addEdge(START, 'agent');
  workflow.addEdge('agent', END);

  return workflow.compile({ 
    checkpointer,
    interruptBefore: [],
    interruptAfter: []
  });
}

export class ShippingAgentExecutor {
  constructor(agent, checkpointer) {
    this.agent = agent;
    this.checkpointer = checkpointer;
  }

  async invoke(state, config) {
    try {
      const result = await this.agent.invoke(state, config);
      return result;
    } catch (error) {
      console.error('Agent execution error:', error);
      throw error;
    }
  }
}

async function generateShippingQuote(shipmentData) {
  const { cargo, weight, serviceLevel, specialRequirements, declaredValue, origin, destination } = shipmentData;
  
  // Extract weight value
  const weightMatch = (weight || cargo || '').match(/(\d+)\s*kg/i);
  const weightValue = weightMatch ? parseInt(weightMatch[1]) : 50;
  
  // Extract declared value
  const valueMatch = (declaredValue || '').match(/[\d,]+/);
  const value = valueMatch ? parseInt(valueMatch[0].replace(/,/g, '')) : 1000;
  
  const routeType = determineRouteType(origin, destination);
  const baseRate = calculateBaseRate(routeType, weightValue, value);
  const service = getServiceMultiplier(serviceLevel);
  const additionalCost = calculateAdditionalCosts(specialRequirements, value);
  
  const carriers = [
    { carrierId: 'dhl_express_001', name: 'DHL Express Worldwide', reputation: 9.4, reliability: 98.7 },
    { carrierId: 'fedex_intl_002', name: 'FedEx International Premium', reputation: 9.2, reliability: 98.2 },
    { carrierId: 'ups_worldwide_003', name: 'UPS Worldwide Express', reputation: 9.0, reliability: 97.8 },
    { carrierId: 'maersk_004', name: 'Maersk Line Freight', reputation: 8.9, reliability: 97.5 },
    { carrierId: 'db_schenker_005', name: 'DB Schenker Global', reputation: 8.8, reliability: 97.2 }
  ];
  
  const quotes = carriers.slice(0, 3).map((carrier, index) => {
    const variation = 0.88 + (index * 0.08);
    const finalRate = (baseRate * service.multiplier * variation + additionalCost);
    const baseDays = service.days.split('-').map(d => parseInt(d));
    const minDays = baseDays[0] + index;
    const maxDays = baseDays[1] + index;
    
    return {
      carrierId: carrier.carrierId,
      name: carrier.name,
      service: serviceLevel || 'Standard',
      rate: finalRate.toFixed(2),
      transitTime: `${minDays}-${maxDays} business days`,
      reputation: carrier.reputation,
      reliability: carrier.reliability + '%',
      estimatedDelivery: new Date(Date.now() + (minDays + 1) * 24 * 60 * 60 * 1000).toISOString(),
      currency: 'USD'
    };
  });
  
  quotes.sort((a, b) => parseFloat(a.rate) - parseFloat(b.rate));
  
  return {
    quotes,
    recommendedQuote: quotes[0],
    totalEstimate: quotes[0].rate,
    currency: 'USD',
    validUntil: new Date(Date.now() + 48 * 60 * 60 * 1000).toISOString(),
    shipmentDetails: {
      weight: weightValue + 'kg',
      declaredValue: '$' + value.toLocaleString(),
      route: `${origin || 'Origin'} → ${destination || 'Destination'}`,
      serviceLevel: serviceLevel || 'Standard',
      cargo: cargo || 'General cargo'
    }
  };
}

function determineRouteType(origin, destination) {
  if (!origin || !destination) return 'domestic';
  
  const originLower = origin.toLowerCase();
  const destLower = destination.toLowerCase();
  
  const countries = ['usa', 'us', 'united states', 'uk', 'united kingdom', 'india', 'china', 
                     'japan', 'germany', 'france', 'canada', 'australia', 'brazil', 'mexico'];
  
  const originCountry = countries.find(c => originLower.includes(c));
  const destCountry = countries.find(c => destLower.includes(c));
  
  if (originCountry && destCountry && originCountry !== destCountry) {
    return 'international';
  }
  
  if (originLower.includes('asia') || destLower.includes('asia') ||
      originLower.includes('europe') || destLower.includes('europe')) {
    return 'international';
  }
  
  return 'domestic';
}

function calculateBaseRate(routeType, weight, value) {
  const routes = { 
    domestic: 120, 
    regional: 280, 
    international: 480 
  };
  const baseRate = routes[routeType] || routes.domestic;
  const weightRate = Math.ceil(weight / 10) * 18;
  const valueRate = Math.ceil(value / 1000) * 5;
  return baseRate + weightRate + valueRate;
}

function getServiceMultiplier(serviceLevel) {
  const multipliers = {
    Express: { multiplier: 2.5, days: '1-3' },
    Standard: { multiplier: 1.0, days: '4-7' },
    Economy: { multiplier: 0.75, days: '8-14' }
  };
  return multipliers[serviceLevel] || multipliers.Standard;
}

function calculateAdditionalCosts(requirements, value) {
  if (!requirements) return 0;
  let cost = 0;
  const reqString = Array.isArray(requirements) 
    ? requirements.join(' ').toLowerCase() 
    : String(requirements).toLowerCase();
  
  if (reqString.includes('insurance')) cost += Math.max(60, value * 0.008);
  if (reqString.includes('fragile') || reqString.includes('handle with care')) cost += 35;
  if (reqString.includes('hazardous') || reqString.includes('dangerous')) cost += 150;
  if (reqString.includes('temperature') || reqString.includes('refrigerated')) cost += 95;
  if (reqString.includes('customs') || reqString.includes('clearance')) cost += 55;
  if (reqString.includes('express') || reqString.includes('urgent')) cost += 75;
  
  return cost;
}