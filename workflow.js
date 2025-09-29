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
      content: "Hello! I'm your AI shipping agent. I'll help you get the best freight quotes. To start, could you tell me where you're shipping from and to? (Example: From Mumbai, India to New York, USA)",
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

  // User has sent a message - process it
  const userMessage = lastMessage.content;
  
  // Build conversation context
  const conversationHistory = state.messages.slice(-6).map(m => 
    m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
  );

  // System prompt for the agent
  const systemPrompt = `You are a professional freight shipping assistant helping users book shipments.

Current Phase: ${state.currentPhase}
Collected Data: ${JSON.stringify(state.shipmentData, null, 2)}

Your task:
1. Extract information from user messages (origin, destination, cargo details, weight, service level, etc.)
2. Guide them through collecting: route, cargo details, service preference, special requirements, declared value, contact info
3. Once you have enough info (at minimum: origin, destination, cargo, weight), generate a quote

Respond naturally and conversationally. Ask for ONE piece of information at a time.

If you have collected: origin, destination, cargo (with approximate weight), then respond with:
GENERATE_QUOTE: {extracted data as JSON}

Otherwise, continue the conversation by asking for the next piece of information.`;

  try {
    const response = await model.invoke([
      new SystemMessage(systemPrompt),
      ...conversationHistory
    ]);

    let assistantResponse = response.content;
    let shouldGenerateQuote = false;
    let extractedData = state.shipmentData;

    // Check if agent wants to generate quote
    if (assistantResponse.includes('GENERATE_QUOTE:')) {
      shouldGenerateQuote = true;
      const jsonMatch = assistantResponse.match(/GENERATE_QUOTE:\s*({[\s\S]*})/);
      if (jsonMatch) {
        try {
          extractedData = { ...state.shipmentData, ...JSON.parse(jsonMatch[1]) };
        } catch (e) {
          console.log('Failed to parse quote data');
        }
      }
    } else {
      // Extract data from user message using AI
      const extractionPrompt = `Extract shipping information from this message: "${userMessage}"

Current data: ${JSON.stringify(state.shipmentData, null, 2)}

Return JSON with any found fields:
{
  "origin": "city, country",
  "destination": "city, country", 
  "cargo": "description",
  "weight": "number with unit",
  "serviceLevel": "Express|Standard|Economy",
  "specialRequirements": "description",
  "declaredValue": "amount",
  "contactName": "name",
  "contactEmail": "email",
  "contactPhone": "phone"
}

Only include fields with information. Return valid JSON only.`;

      try {
        const extractResponse = await model.invoke([
          new SystemMessage(extractionPrompt),
          new HumanMessage(userMessage)
        ]);

        const extracted = JSON.parse(extractResponse.content);
        extractedData = { ...state.shipmentData, ...extracted };
      } catch (e) {
        // Extraction failed, keep existing data
      }
    }

    // Generate quote if ready
    if (shouldGenerateQuote) {
      const quote = await generateShippingQuote(extractedData);
      
      const quoteMessage = `Great! I've found the best shipping options for you:

**Top 3 Carriers:**

1. **${quote.quotes[0].name}** - $${quote.quotes[0].rate}
   - Transit: ${quote.quotes[0].transitTime}
   - Reliability: ${quote.quotes[0].reliability}

2. **${quote.quotes[1].name}** - $${quote.quotes[1].rate}
   - Transit: ${quote.quotes[1].transitTime}
   - Reliability: ${quote.quotes[1].reliability}

3. **${quote.quotes[2].name}** - $${quote.quotes[2].rate}
   - Transit: ${quote.quotes[2].transitTime}
   - Reliability: ${quote.quotes[2].reliability}

📦 **Shipment Details:**
- Route: ${extractedData.origin || 'Origin'} → ${extractedData.destination || 'Destination'}
- Weight: ${extractedData.weight || 'N/A'}
- Service: ${extractedData.serviceLevel || 'Standard'}

Would you like to book one of these options?`;

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
    const errorMessage = "I apologize, I encountered an error. Could you please repeat that?";
    
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
  if (data.origin && data.destination && data.cargo) return 'finalizing';
  if (data.cargo) return 'service_selection';
  if (data.origin || data.destination) return 'cargo_collection';
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
  const { cargo, serviceLevel, specialRequirements, declaredValue, origin, destination } = shipmentData;
  
  const weightMatch = cargo?.match(/(\d+)\s*kg/i);
  const weight = weightMatch ? parseInt(weightMatch[1]) : 50;
  
  const valueMatch = declaredValue?.match(/[\d,]+/);
  const value = valueMatch ? parseInt(valueMatch[0].replace(/,/g, '')) : 1000;
  
  const routeType = determineRouteType(origin, destination);
  const baseRate = calculateBaseRate(routeType, weight, value);
  const service = getServiceMultiplier(serviceLevel);
  const additionalCost = calculateAdditionalCosts(specialRequirements, value);
  
  const carriers = [
    { carrierId: 'dhl_express_001', name: 'DHL Express Worldwide', reputation: 9.4, reliability: 98.7 },
    { carrierId: 'fedex_intl_002', name: 'FedEx International Premium', reputation: 9.2, reliability: 98.2 },
    { carrierId: 'ups_worldwide_003', name: 'UPS Worldwide Express', reputation: 9.0, reliability: 97.8 }
  ];
  
  const quotes = carriers.map((carrier, index) => {
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
      transitTime: `${minDays}-${maxDays} days`,
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
      weight: weight + 'kg',
      declaredValue: '$' + value.toLocaleString(),
      route: `${origin || 'Origin'} → ${destination || 'Destination'}`,
      serviceLevel: serviceLevel || 'Standard'
    }
  };
}

function determineRouteType(origin, destination) {
  if (!origin || !destination) return 'domestic';
  const international = ['usa', 'europe', 'china', 'japan', 'uk', 'canada'];
  const isInternational = international.some(c => 
    origin.toLowerCase().includes(c) || destination.toLowerCase().includes(c)
  );
  return isInternational ? 'international' : 'domestic';
}

function calculateBaseRate(routeType, weight, value) {
  const routes = { domestic: 120, regional: 280, international: 480 };
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
  const reqString = Array.isArray(requirements) ? requirements.join(' ').toLowerCase() : requirements.toLowerCase();
  if (reqString.includes('insurance')) cost += Math.max(60, value * 0.008);
  if (reqString.includes('fragile')) cost += 35;
  if (reqString.includes('hazardous')) cost += 150;
  if (reqString.includes('temperature')) cost += 95;
  if (reqString.includes('customs')) cost += 55;
  return cost;
}