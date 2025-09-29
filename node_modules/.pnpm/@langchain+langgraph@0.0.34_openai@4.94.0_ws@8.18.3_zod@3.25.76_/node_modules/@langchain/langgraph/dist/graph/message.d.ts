import { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import { StateGraph } from "./state.js";
type Messages = Array<BaseMessage | BaseMessageLike> | BaseMessage | BaseMessageLike;
export declare function messagesStateReducer(left: BaseMessage[], right: Messages): BaseMessage[];
export declare class MessageGraph extends StateGraph<BaseMessage[], BaseMessage[], Messages> {
    constructor();
}
export interface MessagesState {
    messages: BaseMessage[];
}
export {};
