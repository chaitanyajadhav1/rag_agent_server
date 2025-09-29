"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MessageGraph = exports.messagesStateReducer = void 0;
const messages_1 = require("@langchain/core/messages");
const uuid_1 = require("uuid");
const state_js_1 = require("./state.cjs");
function messagesStateReducer(left, right) {
    const leftArray = Array.isArray(left) ? left : [left];
    const rightArray = Array.isArray(right) ? right : [right];
    // coerce to message
    const leftMessages = leftArray.map(messages_1.coerceMessageLikeToMessage);
    const rightMessages = rightArray.map(messages_1.coerceMessageLikeToMessage);
    // assign missing ids
    for (const m of leftMessages) {
        if (m.id === null || m.id === undefined) {
            m.id = (0, uuid_1.v4)();
        }
    }
    for (const m of rightMessages) {
        if (m.id === null || m.id === undefined) {
            m.id = (0, uuid_1.v4)();
        }
    }
    // merge
    const leftIdxById = new Map(leftMessages.map((m, i) => [m.id, i]));
    const merged = [...leftMessages];
    const idsToRemove = new Set();
    for (const m of rightMessages) {
        const existingIdx = leftIdxById.get(m.id);
        if (existingIdx !== undefined) {
            if (m._getType() === "remove") {
                idsToRemove.add(m.id);
            }
            else {
                merged[existingIdx] = m;
            }
        }
        else {
            if (m._getType() === "remove") {
                throw new Error(`Attempting to delete a message with an ID that doesn't exist ('${m.id}')`);
            }
            merged.push(m);
        }
    }
    return merged.filter((m) => !idsToRemove.has(m.id));
}
exports.messagesStateReducer = messagesStateReducer;
class MessageGraph extends state_js_1.StateGraph {
    constructor() {
        super({
            channels: {
                __root__: {
                    reducer: messagesStateReducer,
                    default: () => [],
                },
            },
        });
    }
}
exports.MessageGraph = MessageGraph;
